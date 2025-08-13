import argparse
import os
from hybrid_breakhis_core import BreakHisAnalyzer, set_seed


def parse_args():
	parser = argparse.ArgumentParser(description='Train Hybrid BreakHis on Kaggle (multi-GPU ready)')
	parser.add_argument('--data_dir', type=str, required=True, help='Path to BreakHis dataset root')
	parser.add_argument('--mag', type=str, default='400X', choices=['40X','100X','200X','400X'])
	parser.add_argument('--epochs', type=int, default=30)
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--batch_size', type=int, default=24)
	parser.add_argument('--fusion', type=str, default='attention_weighted', choices=['concatenate','attention_weighted','separate_then_combine'])
	parser.add_argument('--workers', type=int, default=2)
	parser.add_argument('--patience', type=int, default=10)
	return parser.parse_args()


def main():
	args = parse_args()
	set_seed(42)
	analyzer = BreakHisAnalyzer(args.data_dir)
	print(f'CUDA available: {os.environ.get("CUDA_VISIBLE_DEVICES", "auto")}; GPUs: {os.popen("nvidia-smi -L").read()}')
	print(f'Loading dataset from: {args.data_dir} at {args.mag}')
	counts = analyzer.load_dataset(args.mag, extract_morphological=True)
	analyzer.split_dataset()
	print('Class distribution:', counts)
	best_acc = analyzer.train_model(
		epochs=args.epochs,
		learning_rate=args.lr,
		fusion_strategy=args.fusion,
		efficientnet_version='b4',
		batch_size=args.batch_size,
		num_workers=args.workers,
		use_weighted_sampler=True,
		use_amp=True,
		label_smoothing=0.1,
		patience=args.patience,
		mixup_alpha=0.2,
		mixup_prob=0.3,
		clip_grad_norm=1.0
	)
	print(f'Best validation accuracy: {best_acc:.2f}%')
	print('Saved best model to best_hybrid_breakhis_model.pth')


if __name__ == '__main__':
	main()