"""Data loaders for EEG datasets (DREAMER and synthetic).

Provides functions to load and preprocess EEG data for training EEGNet models.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES = ["focused", "relaxed", "stressed", "drowsy"]


def load_dreamer(
    mat_path: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load and preprocess the DREAMER dataset.

    Extracts EEG stimuli data for all subjects/trials, windows into 1-second
    epochs at 128Hz with 50% overlap, and creates 4-class labels from
    valence/arousal ratings.

    Label mapping:
        0 (focused):  high valence + high arousal
        1 (relaxed):  high valence + low arousal
        2 (stressed): low valence + high arousal
        3 (drowsy):   low valence + low arousal

    Args:
        mat_path: Path to DREAMER.mat file.

    Returns:
        Tuple of (X, y, class_names) where:
            X: np.ndarray shape (n_epochs, 14, 128)
            y: np.ndarray shape (n_epochs,)
            class_names: list of class name strings
    """
    from scipy.io import loadmat

    logger.info("Loading DREAMER dataset from %s", mat_path)
    mat = loadmat(mat_path, squeeze_me=True)

    dreamer = mat["DREAMER"]
    # DREAMER structure: Data -> subject -> stimuli -> EEG data, valence, arousal
    data_field = dreamer["Data"].item()

    all_epochs = []
    all_labels = []

    n_subjects = len(data_field)
    sfreq = 128  # DREAMER sampling rate
    epoch_len = sfreq  # 1 second = 128 samples
    overlap = epoch_len // 2  # 50% overlap

    for subj_idx in range(n_subjects):
        subject = data_field[subj_idx]
        eeg_stimuli = subject["EEG"].item()["stimuli"].item()
        valence_scores = subject["ScoreValence"].item()
        arousal_scores = subject["ScoreArousal"].item()

        n_trials = len(eeg_stimuli)
        for trial_idx in range(n_trials):
            # Get valence and arousal for this trial
            valence = float(valence_scores[trial_idx])
            arousal = float(arousal_scores[trial_idx])

            # Skip neutral ratings (score == 3)
            if valence == 3.0 or arousal == 3.0:
                continue

            # Binarize: 1-2 = low, 4-5 = high
            high_valence = valence > 3.0
            high_arousal = arousal > 3.0

            # 4-class label
            if high_valence and high_arousal:
                label = 0  # focused
            elif high_valence and not high_arousal:
                label = 1  # relaxed
            elif not high_valence and high_arousal:
                label = 2  # stressed
            else:
                label = 3  # drowsy

            # Get EEG data for this trial: (n_samples, 14)
            eeg_trial = np.array(eeg_stimuli[trial_idx], dtype=np.float64)
            if eeg_trial.ndim == 1:
                continue  # skip malformed trials
            # Transpose to (14, n_samples) if needed
            if eeg_trial.shape[1] == 14 and eeg_trial.shape[0] != 14:
                eeg_trial = eeg_trial.T

            n_samples = eeg_trial.shape[1]

            # Window into epochs with 50% overlap
            start = 0
            while start + epoch_len <= n_samples:
                epoch = eeg_trial[:, start:start + epoch_len]
                all_epochs.append(epoch)
                all_labels.append(label)
                start += overlap

    if not all_epochs:
        raise ValueError("No valid epochs extracted from DREAMER dataset")

    X = np.array(all_epochs, dtype=np.float64)
    y = np.array(all_labels, dtype=np.int64)

    logger.info(
        "DREAMER loaded: %d epochs, %d channels, %d timepoints, %d classes",
        X.shape[0], X.shape[1], X.shape[2], len(CLASS_NAMES),
    )
    for i, name in enumerate(CLASS_NAMES):
        count = int(np.sum(y == i))
        logger.info("  Class %d (%s): %d epochs", i, name, count)

    return X, y, CLASS_NAMES


def generate_synthetic_eeg(
    n_epochs: int = 500,
    n_chans: int = 14,
    n_times: int = 128,
    n_classes: int = 4,
    sfreq: int = 128,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic EEG data with class-specific frequency patterns.

    Each class has a distinct dominant frequency band:
        Class 0 (focused):  elevated beta (13-30 Hz)
        Class 1 (relaxed):  elevated alpha (8-13 Hz)
        Class 2 (stressed): high frequency noise + gamma (30-45 Hz)
        Class 3 (drowsy):   elevated theta (4-8 Hz)

    Args:
        n_epochs: Number of epochs to generate.
        n_chans: Number of EEG channels.
        n_times: Number of time samples per epoch.
        n_classes: Number of classes.
        sfreq: Sampling frequency in Hz.
        rng_seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) where:
            X: np.ndarray shape (n_epochs, n_chans, n_times)
            y: np.ndarray shape (n_epochs,)
    """
    rng = np.random.default_rng(rng_seed)
    t = np.arange(n_times) / sfreq

    X = np.zeros((n_epochs, n_chans, n_times), dtype=np.float64)
    y = np.zeros(n_epochs, dtype=np.int64)

    # Class frequency profiles: (dominant_freq_range_low, dominant_freq_range_high, amplitude_scale)
    class_profiles = {
        0: (15.0, 25.0, 2.0),   # beta - focused
        1: (9.0, 12.0, 2.5),    # alpha - relaxed
        2: (32.0, 42.0, 1.5),   # gamma - stressed
        3: (5.0, 7.0, 2.5),     # theta - drowsy
    }

    epochs_per_class = n_epochs // n_classes
    remainder = n_epochs % n_classes

    idx = 0
    for cls in range(n_classes):
        count = epochs_per_class + (1 if cls < remainder else 0)
        freq_low, freq_high, amp_scale = class_profiles[cls]

        for _ in range(count):
            for ch in range(n_chans):
                # Background noise (pink-ish)
                noise = rng.normal(0, 1.0, n_times)

                # Dominant frequency component
                freq = rng.uniform(freq_low, freq_high)
                phase = rng.uniform(0, 2 * np.pi)
                signal = amp_scale * np.sin(2 * np.pi * freq * t + phase)

                # Add some secondary frequencies for realism
                secondary_freq = rng.uniform(1.0, 45.0)
                secondary_signal = 0.5 * np.sin(2 * np.pi * secondary_freq * t + rng.uniform(0, 2 * np.pi))

                X[idx, ch, :] = noise + signal + secondary_signal

            y[idx] = cls
            idx += 1

    # Shuffle
    perm = rng.permutation(n_epochs)
    X = X[perm]
    y = y[perm]

    logger.info("Generated synthetic EEG: %d epochs, %d channels, %d timepoints", n_epochs, n_chans, n_times)
    return X, y
