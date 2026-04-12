"""Train an EEGNet model for EEG-based brain state classification.

Uses braindecode's EEGNet architecture with a pure PyTorch training loop.
Supports both the DREAMER dataset and synthetic data for pipeline validation.

Usage:
    python -m src.train_eegnet --input DREAMER.mat --output models/eegnet_emotion.pt
    python -m src.train_eegnet --synthetic --output models/eegnet_synthetic.pt
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)


def bandpass_filter(
    data: np.ndarray,
    low: float = 4.0,
    high: float = 45.0,
    sfreq: int = 128,
    order: int = 4,
) -> np.ndarray:
    """Apply a bandpass filter to EEG data.

    Args:
        data: EEG array of shape (n_epochs, n_chans, n_times) or (n_chans, n_times).
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.
        sfreq: Sampling frequency in Hz.
        order: Filter order.

    Returns:
        Filtered data with the same shape.
    """
    sos = butter(order, [low, high], btype="band", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Z-score normalize per channel.

    Args:
        data: Shape (n_epochs, n_chans, n_times).

    Returns:
        Normalized data with the same shape.
    """
    means = data.mean(axis=-1, keepdims=True)
    stds = data.std(axis=-1, keepdims=True)
    stds[stds < 1e-8] = 1.0
    return (data - means) / stds


def train(
    X: np.ndarray,
    y: np.ndarray,
    n_outputs: int,
    output_path: str,
    n_epochs_train: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """Train an EEGNet model and save to disk.

    Args:
        X: Input data shape (n_samples, n_chans, n_times).
        y: Labels shape (n_samples,).
        n_outputs: Number of output classes.
        output_path: Path to save the .pt model file.
        n_epochs_train: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Adam learning rate.
        test_size: Fraction of data for testing.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with training metadata (accuracy, etc.).
    """
    try:
        import torch
        from braindecode.models import EEGNet
    except ImportError:
        logger.error(
            "Deep learning dependencies not installed. "
            "Install with: pip install -e '.[deep]'"
        )
        sys.exit(1)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    n_samples, n_chans, n_times = X.shape
    print(f"Data: {n_samples} samples, {n_chans} channels, {n_times} timepoints, {n_outputs} classes")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train/test split
    indices = np.random.permutation(n_samples)
    split = int(n_samples * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.long)

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Create model
    model = EEGNet(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
    )
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(n_epochs_train):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == n_epochs_train - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(X_test)
                test_preds = test_logits.argmax(dim=1)
                test_acc = (test_preds == y_test).float().mean().item()
            print(f"Epoch {epoch + 1:3d}/{n_epochs_train}: loss={avg_loss:.4f}, test_acc={test_acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_preds = test_logits.argmax(dim=1).numpy()
        y_test_np = y_test.numpy()

    overall_acc = float(np.mean(test_preds == y_test_np))
    print(f"\nFinal test accuracy: {overall_acc:.4f}")

    # Per-class accuracy
    per_class_acc = {}
    for cls in range(n_outputs):
        mask = y_test_np == cls
        if mask.sum() > 0:
            cls_acc = float(np.mean(test_preds[mask] == y_test_np[mask]))
            per_class_acc[cls] = cls_acc
            print(f"  Class {cls}: {cls_acc:.4f} ({mask.sum()} samples)")

    # Save model
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\nModel saved to {output_path}")

    # Save metadata alongside
    metadata = {
        "n_chans": n_chans,
        "n_times": n_times,
        "n_outputs": n_outputs,
        "class_names": ["focused", "relaxed", "stressed", "drowsy"][:n_outputs],
        "accuracy": round(overall_acc, 4),
        "per_class_accuracy": {str(k): round(v, 4) for k, v in per_class_acc.items()},
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_epochs_trained": n_epochs_train,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    }
    meta_path = output_path.replace(".pt", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an EEGNet model for brain state classification.",
    )
    parser.add_argument(
        "--input",
        type=str,
        metavar="MAT_FILE",
        default=None,
        help="Path to DREAMER.mat file",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for pipeline testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the trained model (.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=500,
        help="Number of synthetic epochs to generate (default: 500)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.input and not args.synthetic:
        parser.error("Must provide --input MAT_FILE or --synthetic")

    from .data_loader import CLASS_NAMES, generate_synthetic_eeg, load_dreamer

    if args.input:
        X, y, class_names = load_dreamer(args.input)
    else:
        print("Generating synthetic EEG data...")
        X, y = generate_synthetic_eeg(n_epochs=args.n_synthetic)
        class_names = CLASS_NAMES

    n_outputs = len(set(class_names))

    # Preprocess: bandpass filter + z-score
    print("Preprocessing: bandpass filter (4-45 Hz) + z-score normalization...")
    sfreq = 128
    X = bandpass_filter(X, low=4.0, high=45.0, sfreq=sfreq)
    X = zscore_normalize(X)

    # Train
    train(
        X=X,
        y=y,
        n_outputs=n_outputs,
        output_path=args.output,
        n_epochs_train=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
