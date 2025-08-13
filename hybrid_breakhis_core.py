import os
import glob
import random
import warnings
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from skimage import measure
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from efficientnet_pytorch import EfficientNet
warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True


class MacenkoStainNormalizer:
	def __init__(self):
		self.target_stains = np.array([[0.5626, 0.2159],
									   [0.7201, 0.8012],
									   [0.4062, 0.5581]])
		self.target_concentrations = np.array([[1.9705, 1.0308]])
		
	def rgb_to_od(self, img):
		img = np.maximum(img, 1e-6)
		return -np.log(img / 255.0)
	
	def od_to_rgb(self, od):
		rgb = 255 * np.exp(-od)
		return np.clip(rgb, 0, 255).astype(np.uint8)
	
	def get_stain_matrix(self, od, beta=0.15, alpha=1):
		od_flat = od.reshape(-1, 3)
		od_hat = od_flat[(od_flat > beta).any(axis=1)]
		if len(od_hat) == 0:
			return self.target_stains
		cov = np.cov(od_hat.T)
		eigenvals, eigenvecs = np.linalg.eigh(cov)
		idx = np.argsort(eigenvals)[::-1]
		eigenvecs = eigenvecs[:, idx]
		proj = np.dot(od_hat, eigenvecs[:, :2])
		phi = np.arctan2(proj[:, 1], proj[:, 0])
		min_phi = np.percentile(phi, alpha)
		max_phi = np.percentile(phi, 100 - alpha)
		v1 = np.dot(eigenvecs[:, :2], [np.cos(min_phi), np.sin(min_phi)])
		v2 = np.dot(eigenvecs[:, :2], [np.cos(max_phi), np.sin(max_phi)])
		if v1[0] > 0:
			v1 = -v1
		if v2[0] > 0:
			v2 = -v2
		v1 = v1 / np.linalg.norm(v1)
		v2 = v2 / np.linalg.norm(v2)
		return np.column_stack([v1, v2])
	
	def separate_stains(self, od, stain_matrix):
		od_flat = od.reshape(-1, 3)
		concentrations = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)[0]
		concentrations = np.maximum(concentrations, 0)
		return concentrations.T.reshape(od.shape[:2] + (2,))
	
	def normalize_he(self, image):
		od = self.rgb_to_od(image)
		source_stains = self.get_stain_matrix(od)
		concentrations = self.separate_stains(od, source_stains)
		normalized_od = np.dot(concentrations, self.target_stains.T)
		return self.od_to_rgb(normalized_od)


class ImageProcessor:
	@staticmethod
	def create_binary_mask(image, method='otsu'):
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		if method == 'otsu':
			_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		elif method == 'adaptive':
			binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		else:
			_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
		kernel = np.ones((3, 3), np.uint8)
		binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
		binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
		return binary
	
	@staticmethod
	def extract_features(image, mask):
		props = measure.regionprops(mask.astype(int), intensity_image=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
		if len(props) == 0:
			return np.zeros(7)
		largest_region = max(props, key=lambda x: x.area)
		features = [
			largest_region.area,
			largest_region.perimeter,
			largest_region.eccentricity,
			largest_region.solidity,
			largest_region.extent,
			largest_region.mean_intensity,
			(largest_region.perimeter ** 2) / (4 * np.pi * largest_region.area) if largest_region.area > 0 else 0
		]
		return np.array(features, dtype=np.float32)


class BreakHisDataset(Dataset):
	def __init__(self, image_paths, labels, morphological_features=None, transform=None, normalizer=None):
		self.image_paths = image_paths
		self.labels = labels
		self.morphological_features = morphological_features
		self.transform = transform
		self.normalizer = normalizer
		
	def __len__(self):
		return len(self.image_paths)
	
	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.normalizer:
			image = self.normalizer.normalize_he(image)
		if self.transform:
			image = self.transform(image)
		else:
			image = transforms.ToTensor()(Image.fromarray(image))
			image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
		label = self.labels[idx]
		if self.morphological_features is not None:
			morph_features = torch.tensor(self.morphological_features[idx], dtype=torch.float32)
			return image, morph_features, label
		return image, label


class AttentionBlock(nn.Module):
	def __init__(self, F_g, F_l, F_int):
		super(AttentionBlock, self).__init__()
		self.W_g = nn.Sequential(
			nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(F_int)
		)
		self.W_x = nn.Sequential(
			nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(F_int)
		)
		self.psi = nn.Sequential(
			nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
		)
		self.relu = nn.ReLU(inplace=True)
		
	def forward(self, g, x):
		g1 = self.W_g(g)
		x1 = self.W_x(x)
		psi = self.relu(g1 + x1)
		psi = self.psi(psi)
		return x * psi


class HybridBreakHisClassifier(nn.Module):
	def __init__(self, num_classes=8, num_morphological_features=7, efficientnet_version='b4', fusion_strategy='attention_weighted'):
		super(HybridBreakHisClassifier, self).__init__()
		self.num_morphological_features = num_morphological_features
		self.fusion_strategy = fusion_strategy
		self.backbone = EfficientNet.from_pretrained(f'efficientnet-{efficientnet_version}')
		feature_dims = {
			'b0': [32, 24, 40, 112, 1280],
			'b1': [32, 24, 40, 112, 1280],
			'b2': [32, 24, 48, 120, 1408],
			'b3': [40, 32, 48, 136, 1536],
			'b4': [48, 32, 56, 160, 1792]
		}
		dims = feature_dims.get(efficientnet_version, feature_dims['b4'])
		self.cnn_feature_dim = dims[4]
		self.up_conv4 = nn.ConvTranspose2d(dims[4], dims[3], kernel_size=2, stride=2)
		self.att4 = AttentionBlock(dims[3], dims[3], dims[3]//2)
		self.conv4 = self._make_conv_block(dims[3] + dims[3], dims[3])
		self.up_conv3 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2)
		self.att3 = AttentionBlock(dims[2], dims[2], dims[2]//2)
		self.conv3 = self._make_conv_block(dims[2] + dims[2], dims[2])
		self.up_conv2 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2)
		self.att2 = AttentionBlock(dims[1], dims[1], dims[1]//2)
		self.conv2 = self._make_conv_block(dims[1] + dims[1], dims[1])
		self.up_conv1 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2)
		self.att1 = AttentionBlock(dims[0], dims[0], dims[0]//2)
		self.conv1 = self._make_conv_block(dims[0] + dims[0], dims[0])
		self.cnn_feature_extractor = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Flatten(),
			nn.Linear(self.cnn_feature_dim, 512),
			nn.ReLU(),
			nn.Dropout(0.5)
		)
		self.morph_feature_processor = nn.Sequential(
			nn.Linear(num_morphological_features, 64),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(64, 128),
			nn.ReLU(),
			nn.Dropout(0.3)
		)
		self.feature_attention = nn.Sequential(
			nn.Linear(512 + 128, 256),
			nn.ReLU(),
			nn.Linear(256, 512 + 128),
			nn.Sigmoid()
		)
		if fusion_strategy in ['concatenate', 'attention_weighted']:
			self.classifier = nn.Sequential(
				nn.Linear(512 + 128, 256),
				nn.ReLU(),
				nn.Dropout(0.4),
				nn.Linear(256, 128),
				nn.ReLU(),
				nn.Dropout(0.2),
				nn.Linear(128, num_classes)
			)
		elif fusion_strategy == 'separate_then_combine':
			self.cnn_classifier = nn.Sequential(
				nn.Linear(512, 256),
				nn.ReLU(),
				nn.Dropout(0.3),
				nn.Linear(256, num_classes)
			)
			self.morph_classifier = nn.Sequential(
				nn.Linear(128, 64),
				nn.ReLU(),
				nn.Dropout(0.3),
				nn.Linear(64, num_classes)
			)
			self.combination_weights = nn.Parameter(torch.tensor([0.7, 0.3]))
		
	def _make_conv_block(self, in_channels, out_channels):
		return nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	
	def extract_encoder_features(self, x):
		features = []
		x = self.backbone._conv_stem(x)
		x = self.backbone._bn0(x)
		x = self.backbone._swish(x)
		features.append(x)
		for idx, block in enumerate(self.backbone._blocks):
			x = block(x)
			if idx in [2, 4, 10, 16]:
				features.append(x)
		x = self.backbone._conv_head(x)
		x = self.backbone._bn1(x)
		x = self.backbone._swish(x)
		features.append(x)
		return features
	
	def forward(self, x, morphological_features):
		encoder_features = self.extract_encoder_features(x)
		cnn_features = self.cnn_feature_extractor(encoder_features[-1])
		morph_features = self.morph_feature_processor(morphological_features)
		if self.fusion_strategy == 'concatenate':
			combined_features = torch.cat([cnn_features, morph_features], dim=1)
			output = self.classifier(combined_features)
		elif self.fusion_strategy == 'attention_weighted':
			combined_features = torch.cat([cnn_features, morph_features], dim=1)
			attention_weights = self.feature_attention(combined_features)
			weighted_features = combined_features * attention_weights
			output = self.classifier(weighted_features)
		elif self.fusion_strategy == 'separate_then_combine':
			cnn_output = self.cnn_classifier(cnn_features)
			morph_output = self.morph_classifier(morph_features)
			weights = F.softmax(self.combination_weights, dim=0)
			output = weights[0] * cnn_output + weights[1] * morph_output
		return output


class BreakHisAnalyzer:
	def __init__(self, dataset_path):
		self.dataset_path = dataset_path
		self.normalizer = MacenkoStainNormalizer()
		self.processor = ImageProcessor()
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = None
		self.label_encoder = LabelEncoder()
		self.morph_scaler = StandardScaler()
		self.image_paths = []
		self.labels = []
		self.magnifications = []
		self.morphological_features = []
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.mag_train = None
		self.mag_test = None
		self.morph_train = None
		self.morph_test = None
		self.class_mapping = {
			'adenosis': 'Benign - Adenosis',
			'fibroadenoma': 'Benign - Fibroadenoma',
			'phyllodes_tumor': 'Benign - Phyllodes Tumor',
			'tubular_adenoma': 'Benign - Tubular Adenoma',
			'ductal_carcinoma': 'Malignant - Ductal Carcinoma',
			'lobular_carcinoma': 'Malignant - Lobular Carcinoma',
			'mucinous_carcinoma': 'Malignant - Mucinous Carcinoma',
			'papillary_carcinoma': 'Malignant - Papillary Carcinoma'
		}
		self.val_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		self.train_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomVerticalFlip(),
			transforms.RandomRotation(20),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	
	def extract_all_morphological_features(self, progress_callback=None):
		self.morphological_features = []
		total_images = len(self.image_paths)
		for i, image_path in enumerate(self.image_paths):
			try:
				image = cv2.imread(image_path)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				normalized_image = self.normalizer.normalize_he(image)
				binary_mask = self.processor.create_binary_mask(normalized_image)
				features = self.processor.extract_features(normalized_image, binary_mask)
				self.morphological_features.append(features)
				if progress_callback and (i + 1) % 50 == 0:
					progress_callback(i + 1, total_images)
			except Exception:
				self.morphological_features.append(np.zeros(7, dtype=np.float32))
		self.morphological_features = np.array(self.morphological_features, dtype=np.float32)
	
	def detect_dataset_structure(self):
		structure_info = {
			'magnifications': set(),
			'cancer_types': set(),
			'total_images': 0,
			'structure_type': 'unknown'
		}
		for root, dirs, files in os.walk(self.dataset_path):
			image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
			structure_info['total_images'] += len(image_files)
			for dirname in dirs:
				if 'X' in dirname and any(mag in dirname for mag in ['40', '100', '200', '400']):
					structure_info['magnifications'].add(dirname)
			path_parts = root.lower().split(os.sep)
			for part in path_parts:
				if any(cancer in part for cancer in ['adenosis', 'fibroadenoma', 'phyllodes', 'tubular', 'ductal', 'lobular', 'mucinous', 'papillary']):
					structure_info['cancer_types'].add(part)
		return structure_info
		
	def load_dataset(self, selected_magnification='400X', extract_morphological=True):
		class_counts = {}
		self.image_paths = []
		self.labels = []
		self.magnifications = []
		structure_info = self.detect_dataset_structure()
		if structure_info['total_images'] == 0:
			raise ValueError('No image files found in the dataset directory')
		success = False
		success = self.load_standard_breakhis(selected_magnification, class_counts)
		if not success:
			success = self.load_flexible_structure(selected_magnification, class_counts)
		if not success:
			success = self.load_flat_structure(selected_magnification, class_counts)
		if not success or len(self.image_paths) == 0:
			raise ValueError(f'No images found for magnification {selected_magnification}')
		self.labels = self.label_encoder.fit_transform(self.labels)
		if extract_morphological:
			self.extract_all_morphological_features()
		return class_counts
	
	def load_standard_breakhis(self, selected_magnification, class_counts):
		try:
			possible_structures = [
				os.path.join(self.dataset_path, 'BreaKHis_v1', 'histology_slides', 'breast'),
				os.path.join(self.dataset_path, 'histology_slides', 'breast'),
				os.path.join(self.dataset_path, 'breast'),
				self.dataset_path
			]
			breast_path = None
			for structure in possible_structures:
				if os.path.exists(structure):
					benign_path = os.path.join(structure, 'benign')
					malignant_path = os.path.join(structure, 'malignant')
					if os.path.exists(benign_path) and os.path.exists(malignant_path):
						breast_path = structure
						break
			if not breast_path:
				return False
			for category_path in [os.path.join(breast_path, 'benign'), os.path.join(breast_path, 'malignant')]:
				for cancer_type_folder in os.listdir(category_path):
					cancer_type_path = os.path.join(category_path, cancer_type_folder)
					if not os.path.isdir(cancer_type_path):
						continue
					found_images = False
					for sub_folder in os.listdir(cancer_type_path):
						if sub_folder.startswith('SOB_'):
							subfolder_path = os.path.join(cancer_type_path, sub_folder)
							if os.path.isdir(subfolder_path):
								magnification_path = os.path.join(subfolder_path, selected_magnification)
								if os.path.exists(magnification_path):
									images = self.get_images_from_folder(magnification_path)
									if images:
										self.add_images_to_dataset(images, cancer_type_folder, selected_magnification, class_counts)
										found_images = True
					if not found_images:
						magnification_path = os.path.join(cancer_type_path, selected_magnification)
						if os.path.exists(magnification_path):
							images = self.get_images_from_folder(magnification_path)
							if images:
								self.add_images_to_dataset(images, cancer_type_folder, selected_magnification, class_counts)
								found_images = True
			return len(self.image_paths) > 0
		except Exception:
			return False
	
	def load_flexible_structure(self, selected_magnification, class_counts):
		try:
			for root, dirs, files in os.walk(self.dataset_path):
				if selected_magnification in os.path.basename(root):
					images = self.get_images_from_folder(root)
					if images:
						cancer_type = self.infer_cancer_type_from_path(root)
						if cancer_type:
							self.add_images_to_dataset(images, cancer_type, selected_magnification, class_counts)
				for dirname in dirs:
					if dirname == selected_magnification:
						mag_path = os.path.join(root, dirname)
						images = self.get_images_from_folder(mag_path)
						if images:
							cancer_type = self.infer_cancer_type_from_path(root)
							if cancer_type:
								self.add_images_to_dataset(images, cancer_type, selected_magnification, class_counts)
			return len(self.image_paths) > 0
		except Exception:
			return False
	
	def load_flat_structure(self, selected_magnification, class_counts):
		try:
			all_images = []
			for root, dirs, files in os.walk(self.dataset_path):
				for file in files:
					if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
						all_images.append(os.path.join(root, file))
			filtered = [p for p in all_images if selected_magnification in p or selected_magnification.replace('X', '') in p]
			if not filtered:
				filtered = all_images
			for img_path in filtered:
				cancer_type = self.infer_cancer_type_from_path(img_path)
				if cancer_type:
					self.add_images_to_dataset([img_path], cancer_type, selected_magnification, class_counts)
				else:
					default_type = 'adenosis' if 'benign' in img_path.lower() else 'ductal_carcinoma'
					self.add_images_to_dataset([img_path], default_type, selected_magnification, class_counts)
			return len(self.image_paths) > 0
		except Exception:
			return False
	
	def get_images_from_folder(self, folder_path):
		image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
		images = []
		for ext in image_extensions:
			images.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
			images.extend(glob.glob(os.path.join(folder_path, f'*{ext.upper()}')))
		return images
	
	def infer_cancer_type_from_path(self, path):
		path_lower = path.lower()
		type_keywords = {
			'adenosis': 'adenosis',
			'fibroadenoma': 'fibroadenoma',
			'phyllodes': 'phyllodes_tumor',
			'tubular': 'tubular_adenoma',
			'ductal': 'ductal_carcinoma',
			'lobular': 'lobular_carcinoma',
			'mucinous': 'mucinous_carcinoma',
			'papillary': 'papillary_carcinoma'
		}
		for keyword, cancer_type in type_keywords.items():
			if keyword in path_lower:
				return cancer_type
		return None
	
	def add_images_to_dataset(self, images, cancer_type, magnification, class_counts):
		normalized_type = cancer_type.lower().replace(' ', '_')
		self.image_paths.extend(images)
		self.labels.extend([normalized_type] * len(images))
		self.magnifications.extend([magnification] * len(images))
		class_counts[normalized_type] = class_counts.get(normalized_type, 0) + len(images)
	
	def split_dataset(self, test_size=0.2):
		if len(self.morphological_features) > 0:
			self.X_train, self.X_test, self.y_train, self.y_test, self.mag_train, self.mag_test, morph_train_raw, morph_test_raw = train_test_split(
				self.image_paths, self.labels, self.magnifications, self.morphological_features,
				test_size=test_size, random_state=42, stratify=self.labels
			)
			self.morph_scaler.fit(morph_train_raw)
			self.morph_train = self.morph_scaler.transform(morph_train_raw)
			self.morph_test = self.morph_scaler.transform(morph_test_raw)
		else:
			self.X_train, self.X_test, self.y_train, self.y_test, self.mag_train, self.mag_test = train_test_split(
				self.image_paths, self.labels, self.magnifications,
				test_size=test_size, random_state=42, stratify=self.labels
			)
			self.morph_train = None
			self.morph_test = None
	
	def _compute_sample_weights(self, labels_encoded):
		labels_np = np.array(labels_encoded)
		class_sample_counts = np.bincount(labels_np)
		class_sample_counts = np.maximum(class_sample_counts, 1)
		class_weights = class_sample_counts.sum() / (class_sample_counts.astype(np.float32) * len(class_sample_counts))
		weights = class_weights[labels_np]
		return weights.tolist()
	
	def create_dataloaders(self, batch_size=16, num_workers=None, use_weighted_sampler=True):
		num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 2)
		pin_memory = self.device.type == 'cuda'
		persistent = num_workers > 0
		train_dataset = BreakHisDataset(
			self.X_train, self.y_train,
			morphological_features=self.morph_train,
			transform=self.train_transform,
			normalizer=self.normalizer
		)
		test_dataset = BreakHisDataset(
			self.X_test, self.y_test,
			morphological_features=self.morph_test,
			transform=self.val_transform,
			normalizer=self.normalizer
		)
		if use_weighted_sampler:
			weights = self._compute_sample_weights(self.y_train)
			sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
			train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
		else:
			train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent)
		return train_loader, test_loader
	
	def _build_criterion(self, label_smoothing=0.1, balance_loss=True):
		if balance_loss:
			counts = np.bincount(np.array(self.y_train))
			counts = np.maximum(counts, 1)
			class_weights = counts.sum() / (counts.astype(np.float32) * len(counts))
			weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
			return nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
		return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
	
	def _maybe_wrap_dataparallel(self):
		if torch.cuda.is_available() and torch.cuda.device_count() > 1:
			self.model = nn.DataParallel(self.model)
			print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
	
	def _mixup(self, x, mf, y, alpha=0.2):
		if alpha <= 0:
			return x, mf, y, None, None, 1.0
		lam = np.random.beta(alpha, alpha)
		batch_size = x.size(0)
		index = torch.randperm(batch_size, device=x.device)
		mixed_x = lam * x + (1 - lam) * x[index, :]
		mixed_mf = lam * mf + (1 - lam) * mf[index, :]
		y_a, y_b = y, y[index]
		return mixed_x, mixed_mf, y, y_a, y_b, lam
	
	def train_model(self, epochs=50, learning_rate=0.0002, fusion_strategy='attention_weighted', efficientnet_version='b4', batch_size=24, num_workers=2, use_weighted_sampler=True, use_amp=True, label_smoothing=0.1, patience=10, mixup_alpha=0.2, mixup_prob=0.3, clip_grad_norm=1.0):
		set_seed(42)
		num_classes = len(self.label_encoder.classes_)
		self.model = HybridBreakHisClassifier(
			num_classes=num_classes,
			num_morphological_features=7,
			efficientnet_version=efficientnet_version,
			fusion_strategy=fusion_strategy
		).to(self.device)
		self._maybe_wrap_dataparallel()
		criterion = self._build_criterion(label_smoothing=label_smoothing, balance_loss=True)
		optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
		train_loader, test_loader = self.create_dataloaders(batch_size=batch_size, num_workers=num_workers, use_weighted_sampler=use_weighted_sampler)
		scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.15, div_factor=10.0, final_div_factor=100.0)
		scaler = GradScaler(enabled=use_amp)
		best_accuracy = 0.0
		no_improve_epochs = 0
		for epoch in range(epochs):
			self.model.train()
			train_correct = 0
			train_total = 0
			for batch_data in train_loader:
				if len(batch_data) == 3:
					images, morph_features, labels = batch_data
					images = images.to(self.device, non_blocking=True)
					morph_features = morph_features.to(self.device, non_blocking=True)
					labels = torch.tensor(labels, dtype=torch.long, device=self.device)
				else:
					images, labels = batch_data
					images = images.to(self.device, non_blocking=True)
					labels = torch.tensor(labels, dtype=torch.long, device=self.device)
					morph_features = torch.zeros((images.size(0), 7), device=self.device)
				optimizer.zero_grad(set_to_none=True)
				apply_mix = mixup_alpha > 0 and random.random() < mixup_prob
				if apply_mix:
					mixed_images, mixed_mf, _, y_a, y_b, lam = self._mixup(images, morph_features, labels, alpha=mixup_alpha)
					with autocast(enabled=use_amp):
						outputs = self.model(mixed_images, mixed_mf)
						loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
				else:
					with autocast(enabled=use_amp):
						outputs = self.model(images, morph_features)
						loss = criterion(outputs, labels)
				scaler.scale(loss).backward()
				if clip_grad_norm and clip_grad_norm > 0:
					scaler.unscale_(optimizer)
					nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
				scaler.step(optimizer)
				scaler.update()
				scheduler.step()
				with torch.no_grad():
					_, predicted = outputs.max(1)
					train_total += labels.size(0)
					train_correct += predicted.eq(labels).sum().item()
			train_accuracy = 100. * train_correct / max(train_total, 1)
			self.model.eval()
			test_correct = 0
			test_total = 0
			with torch.no_grad():
				for batch_data in test_loader:
					if len(batch_data) == 3:
						images, morph_features, labels = batch_data
						images = images.to(self.device, non_blocking=True)
						morph_features = morph_features.to(self.device, non_blocking=True)
						labels = torch.tensor(labels, dtype=torch.long, device=self.device)
					else:
						images, labels = batch_data
						images = images.to(self.device, non_blocking=True)
						labels = torch.tensor(labels, dtype=torch.long, device=self.device)
						morph_features = torch.zeros((images.size(0), 7), device=self.device)
					with autocast(enabled=use_amp):
						outputs = self.model(images, morph_features)
						loss = criterion(outputs, labels)
					_, predicted = outputs.max(1)
					test_total += labels.size(0)
					test_correct += predicted.eq(labels).sum().item()
			test_accuracy = 100. * test_correct / max(test_total, 1)
			if test_accuracy > best_accuracy:
				best_accuracy = test_accuracy
				state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
				torch.save({
					'model_state_dict': state_dict,
					'label_encoder': self.label_encoder,
					'morph_scaler': self.morph_scaler,
					'class_mapping': self.class_mapping,
					'num_classes': num_classes,
					'fusion_strategy': fusion_strategy
				}, 'best_hybrid_breakhis_model.pth')
				no_improve_epochs = 0
			else:
				no_improve_epochs += 1
			print(f'Epoch {epoch+1}/{epochs} | Train Acc: {train_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | Best: {best_accuracy:.2f}%')
			if no_improve_epochs >= patience:
				print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
				break
		return best_accuracy
	
	def evaluate_model(self):
		if self.model is None:
			return None
		_, test_loader = self.create_dataloaders()
		self.model.eval()
		y_true, y_pred = [], []
		with torch.no_grad():
			for batch_data in test_loader:
				if len(batch_data) == 3:
					images, morph_features, labels = batch_data
					images = images.to(self.device, non_blocking=True)
					morph_features = morph_features.to(self.device, non_blocking=True)
					labels = torch.tensor(labels, dtype=torch.long, device=self.device)
				else:
					images, labels = batch_data
					images = images.to(self.device, non_blocking=True)
					labels = torch.tensor(labels, dtype=torch.long, device=self.device)
					morph_features = torch.zeros((images.size(0), 7), device=self.device)
				with autocast(enabled=True):
					outputs = self.model(images, morph_features)
					_, predicted = outputs.max(1)
				y_true.extend(labels.cpu().numpy())
				y_pred.extend(predicted.cpu().numpy())
		accuracy = accuracy_score(y_true, y_pred)
		class_names = self.label_encoder.classes_
		report = classification_report(y_true, y_pred, target_names=class_names)
		cm = confusion_matrix(y_true, y_pred)
		return accuracy, report, cm, class_names
	
	def load_model(self, filepath, num_classes=None):
		checkpoint = torch.load(filepath, map_location=self.device)
		if num_classes is None:
			num_classes = checkpoint.get('num_classes', 8)
		fusion_strategy = checkpoint.get('fusion_strategy', 'attention_weighted')
		self.model = HybridBreakHisClassifier(
			num_classes=num_classes,
			num_morphological_features=7,
			fusion_strategy=fusion_strategy
		).to(self.device)
		self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
		self.label_encoder = checkpoint['label_encoder']
		self.morph_scaler = checkpoint['morph_scaler']
		self.class_mapping = checkpoint.get('class_mapping', self.class_mapping)
		self._maybe_wrap_dataparallel()
		return True