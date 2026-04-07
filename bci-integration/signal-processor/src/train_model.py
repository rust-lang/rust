"""Train a Random Forest classifier on recorded BCI session data.

Reads JSONL session files (produced by SessionRecorder), extracts feature
vectors and labels, trains a RandomForestClassifier, and saves the model.

Usage:
    python -m src.train_model --input session1.jsonl session2.jsonl --output models/brain_state_rf.joblib
    python -m src.train_model --generate-synthetic --seconds 30 --output models/brain_state_rf.joblib
"""

import argparse
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Feature vector order -- must match MLClassifier.FEATURE_NAMES
FEATURE_NAMES = (
    "delta", "theta", "alpha", "beta", "gamma",
    "attention", "relaxation", "cognitive_load", "signal_quality",
)

# Valid states for training labels
VALID_STATES = {"focused", "relaxed", "stressed", "drowsy", "meditative", "active"}


def load_sessions(file_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load feature vectors and labels from one or more JSONL session files.

    Args:
        file_paths: Paths to JSONL files produced by SessionRecorder.

    Returns:
        Tuple of (features array [N, 9], labels array [N]).
    """
    features = []
    labels = []

    for path in file_paths:
        logger.info("Reading %s", path)
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at %s:%d", path, line_num)
                    continue

                state = record.get("state", {})
                primary = state.get("primary", "unknown")

                if primary not in VALID_STATES:
                    continue  # skip "unknown" or invalid labels

                band_powers = record.get("band_powers", {})
                scores = record.get("scores", {})
                signal_quality = record.get("signal_quality", 0.0)

                feature_vec = [
                    band_powers.get("delta", 0.0),
                    band_powers.get("theta", 0.0),
                    band_powers.get("alpha", 0.0),
                    band_powers.get("beta", 0.0),
                    band_powers.get("gamma", 0.0),
                    scores.get("attention", 0.0),
                    scores.get("relaxation", 0.0),
                    scores.get("cognitive_load", 0.0),
                    signal_quality,
                ]
                features.append(feature_vec)
                labels.append(primary)

    if not features:
        logger.error("No valid training samples found in input files")
        sys.exit(1)

    return np.array(features), np.array(labels)


def generate_synthetic_session(seconds: int, output_jsonl: str) -> str:
    """Run BrainFlow synthetic board, record a session using heuristic labels.

    Args:
        seconds: How many seconds to record.
        output_jsonl: Path to write the JSONL session file.

    Returns:
        Path to the written JSONL file.
    """
    from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

    from .classifier import HeuristicClassifier
    from .dsp import (
        assess_signal_quality,
        compute_attention,
        compute_band_powers,
        compute_cognitive_load,
        compute_relaxation,
        estimate_artifact_probability,
        sanitize_data,
    )
    from . import config

    logger.info("Generating synthetic session for %d seconds -> %s", seconds, output_jsonl)

    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board.prepare_session()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    board.start_stream()

    classifier = HeuristicClassifier()
    sample_rate = config.SAMPLE_RATE
    window_samples = config.WINDOW_SIZE_SAMPLES
    step_seconds = config.WINDOW_STEP_MS / 1000.0

    # Let the buffer fill
    time.sleep(1.0)

    records = []
    end_time = time.time() + seconds

    while time.time() < end_time:
        data = board.get_current_board_data(window_samples)
        if data.shape[1] < 4:
            time.sleep(step_seconds)
            continue

        eeg_data = data[eeg_channels, :]
        eeg_data = sanitize_data(eeg_data)

        signal_quality = assess_signal_quality(eeg_data)
        artifact_prob = estimate_artifact_probability(eeg_data)
        band_powers = compute_band_powers(eeg_data, sample_rate)
        attention = compute_attention(band_powers)
        relaxation = compute_relaxation(band_powers)
        cog_load = compute_cognitive_load(band_powers)

        result = classifier.classify(
            band_powers=band_powers,
            attention=attention,
            relaxation=relaxation,
            cognitive_load=cog_load,
            signal_quality=signal_quality,
        )

        now_ms = int(time.time() * 1000)
        record = {
            "timestamp_unix_ms": now_ms,
            "session_id": "session-synthetic-train",
            "device_id": f"brainflow-board-{BoardIds.SYNTHETIC_BOARD.value}",
            "state": {
                "primary": result.primary,
                "confidence": result.confidence,
                "secondary": result.secondary,
            },
            "scores": {
                "attention": round(attention, 3),
                "relaxation": round(relaxation, 3),
                "cognitive_load": round(cog_load, 3),
            },
            "band_powers": {k: round(v, 6) for k, v in band_powers.items()},
            "signal_quality": round(signal_quality, 3),
            "artifact_probability": round(artifact_prob, 3),
            "staleness_ms": 0,
            "natural_language_summary": "",
        }
        records.append(record)
        time.sleep(step_seconds)

    board.stop_stream()
    board.release_session()

    # Write JSONL
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    logger.info("Recorded %d samples to %s", len(records), output_jsonl)
    return output_jsonl


def train_and_save(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str,
) -> None:
    """Train a Random Forest model, print metrics, and save to disk.

    Args:
        features: Feature matrix [N, 9].
        labels: Label array [N].
        output_path: Path to save the joblib model file.
    """
    unique_labels = np.unique(labels)
    print(f"Training samples: {len(labels)}")
    print(f"Classes: {list(unique_labels)}")
    print(f"Distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print()

    # If we have very few samples, use a smaller test split
    if len(labels) < 10:
        print("WARNING: Very few samples. Using all data for training (no test split).")
        X_train, y_train = features, labels
        X_test, y_test = features, labels
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels,
        )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, labels=model.classes_, zero_division=0))

    # Save model
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest brain state classifier from recorded sessions.",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        metavar="FILE",
        help="One or more JSONL session files to train from",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the trained model (.joblib)",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate synthetic training data using BrainFlow synthetic board",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=30,
        help="Seconds of synthetic data to generate (default: 30)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.input and not args.generate_synthetic:
        parser.error("Must provide --input files or --generate-synthetic")

    input_files = list(args.input) if args.input else []

    if args.generate_synthetic:
        synthetic_path = os.path.join(
            os.path.dirname(args.output) or ".", "_synthetic_session.jsonl",
        )
        generate_synthetic_session(seconds=args.seconds, output_jsonl=synthetic_path)
        input_files.append(synthetic_path)

    features, labels = load_sessions(input_files)
    train_and_save(features, labels, args.output)


if __name__ == "__main__":
    main()
