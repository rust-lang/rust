"""EEGNet-based deep learning classifier for brain state classification.

Uses braindecode's EEGNet model to classify raw EEG data into emotional/cognitive
states. Requires the 'deep' optional dependencies: torch, braindecode, mne.

Falls back gracefully when deep learning dependencies are not installed.
"""

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.signal import resample

from .classifier import ClassificationResult

logger = logging.getLogger(__name__)

# Label mapping for the 4-class emotion model
DEFAULT_LABEL_MAP: dict[int, str] = {
    0: "focused",
    1: "relaxed",
    2: "stressed",
    3: "drowsy",
}


@runtime_checkable
class RawClassifier(Protocol):
    """Protocol for classifiers that operate on raw EEG data."""

    def classify_raw(self, eeg_data: np.ndarray, sample_rate: int) -> ClassificationResult:
        """Classify from raw EEG data (n_channels, n_samples)."""
        ...


class DeepClassifier:
    """Classifies brain state from raw EEG using an EEGNet model.

    Args:
        model_path: Path to a .pt file containing EEGNet state_dict.
        n_chans: Number of EEG channels the model expects.
        n_times: Number of time samples per window the model expects.
        n_outputs: Number of output classes.
        target_sfreq: Target sampling frequency (Hz) for the model.
    """

    def __init__(
        self,
        model_path: str,
        n_chans: int = 14,
        n_times: int = 128,
        n_outputs: int = 4,
        target_sfreq: int = 128,
    ) -> None:
        self._n_chans = n_chans
        self._n_times = n_times
        self._n_outputs = n_outputs
        self._target_sfreq = target_sfreq
        self._label_map = {i: DEFAULT_LABEL_MAP.get(i, f"class_{i}") for i in range(n_outputs)}
        self._model = None

        try:
            import torch
            from braindecode.models import EEGNet

            self._torch = torch

            # Create model architecture and load weights
            model = EEGNet(
                n_chans=n_chans,
                n_outputs=n_outputs,
                n_times=n_times,
            )
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
            logger.info(
                "Loaded EEGNet model from %s (chans=%d, times=%d, outputs=%d)",
                model_path, n_chans, n_times, n_outputs,
            )
        except FileNotFoundError:
            logger.error("Deep model file not found: %s", model_path)
        except ImportError:
            logger.error(
                "Deep learning dependencies not installed. "
                "Install with: pip install -e '.[deep]'"
            )
        except Exception:
            logger.exception("Failed to load deep model from %s", model_path)

    @property
    def is_available(self) -> bool:
        """Whether the model was loaded successfully."""
        return self._model is not None

    def classify_raw(self, eeg_data: np.ndarray, sample_rate: int) -> ClassificationResult:
        """Classify brain state from raw EEG data.

        Args:
            eeg_data: Raw EEG array of shape (n_channels, n_samples).
            sample_rate: Sampling rate of the input data in Hz.

        Returns:
            ClassificationResult with primary state, confidence, and secondary states.
        """
        if self._model is None:
            return ClassificationResult(primary="unknown", confidence=0.0, secondary=[])

        try:
            import torch

            data = eeg_data.copy().astype(np.float64)

            # Handle channel count mismatch
            n_input_chans = data.shape[0]
            if n_input_chans > self._n_chans:
                # Truncate to expected number of channels
                data = data[:self._n_chans, :]
            elif n_input_chans < self._n_chans:
                # Zero-pad missing channels
                pad = np.zeros((self._n_chans - n_input_chans, data.shape[1]), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)

            # Resample to target sampling frequency if needed
            if sample_rate != self._target_sfreq:
                n_target_samples = int(data.shape[1] * self._target_sfreq / sample_rate)
                if n_target_samples < 1:
                    return ClassificationResult(primary="unknown", confidence=0.0, secondary=[])
                data = resample(data, n_target_samples, axis=1)

            # Take last n_times samples (1 window)
            if data.shape[1] < self._n_times:
                # Zero-pad at the beginning if not enough samples
                pad_width = self._n_times - data.shape[1]
                data = np.pad(data, ((0, 0), (pad_width, 0)), mode="constant")
            else:
                data = data[:, -self._n_times:]

            # Z-score normalize per channel
            means = data.mean(axis=1, keepdims=True)
            stds = data.std(axis=1, keepdims=True)
            stds[stds < 1e-8] = 1.0  # avoid division by zero
            data = (data - means) / stds

            # Convert to tensor: (1, n_chans, n_times)
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

            # Forward pass
            with torch.no_grad():
                logits = self._model(tensor)
                probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()

            # Build result
            ranked = sorted(
                [(self._label_map[i], float(probabilities[i])) for i in range(self._n_outputs)],
                key=lambda x: x[1],
                reverse=True,
            )

            primary_state = ranked[0][0]
            primary_confidence = ranked[0][1]

            secondary = [
                {"state": state, "confidence": round(conf, 3)}
                for state, conf in ranked[1:]
                if conf > 0.05
            ]

            return ClassificationResult(
                primary=primary_state,
                confidence=round(primary_confidence, 3),
                secondary=secondary,
            )

        except Exception:
            logger.exception("Deep classifier inference failed")
            return ClassificationResult(primary="unknown", confidence=0.0, secondary=[])
