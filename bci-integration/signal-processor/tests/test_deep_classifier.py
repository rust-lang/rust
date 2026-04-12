"""Tests for the EEGNet-based deep classifier.

These tests are skipped when torch/braindecode are not installed.
They create a tiny EEGNet model in-memory, save it, load it via
DeepClassifier, and verify inference works correctly.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.classifier import ClassificationResult
from src.deep_classifier import DEFAULT_LABEL_MAP, DeepClassifier, RawClassifier

# Skip all tests in this file if torch or braindecode are not installed
torch = pytest.importorskip("torch")
pytest.importorskip("braindecode")

from braindecode.models import EEGNet  # noqa: E402


# Tiny model config for fast tests
TINY_N_CHANS = 4
TINY_N_TIMES = 64
TINY_N_OUTPUTS = 4
TINY_SFREQ = 64  # 1 second at 64Hz


@pytest.fixture
def tiny_model_path(tmp_path: Path) -> str:
    """Create a tiny untrained EEGNet model, save state_dict, return path."""
    model = EEGNet(
        n_chans=TINY_N_CHANS,
        n_outputs=TINY_N_OUTPUTS,
        n_times=TINY_N_TIMES,
    )
    path = tmp_path / "tiny_eegnet.pt"
    torch.save(model.state_dict(), str(path))
    return str(path)


class TestDeepClassifierBasic:
    def test_protocol_compliance(self, tiny_model_path: str):
        """DeepClassifier should satisfy the RawClassifier protocol."""
        clf = DeepClassifier(
            model_path=tiny_model_path,
            n_chans=TINY_N_CHANS,
            n_times=TINY_N_TIMES,
            n_outputs=TINY_N_OUTPUTS,
            target_sfreq=TINY_SFREQ,
        )
        assert isinstance(clf, RawClassifier)
        assert clf.is_available

    def test_classify_returns_valid_result(self, tiny_model_path: str):
        """classify_raw should return a valid ClassificationResult."""
        clf = DeepClassifier(
            model_path=tiny_model_path,
            n_chans=TINY_N_CHANS,
            n_times=TINY_N_TIMES,
            n_outputs=TINY_N_OUTPUTS,
            target_sfreq=TINY_SFREQ,
        )
        # Input at same sample rate as model
        eeg = np.random.randn(TINY_N_CHANS, TINY_N_TIMES).astype(np.float32) * 20
        result = clf.classify_raw(eeg_data=eeg, sample_rate=TINY_SFREQ)

        assert isinstance(result, ClassificationResult)
        assert result.primary in DEFAULT_LABEL_MAP.values()
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.secondary, list)

    def test_confidence_is_max_probability(self, tiny_model_path: str):
        """Primary confidence should be the highest probability."""
        clf = DeepClassifier(
            model_path=tiny_model_path,
            n_chans=TINY_N_CHANS,
            n_times=TINY_N_TIMES,
            n_outputs=TINY_N_OUTPUTS,
            target_sfreq=TINY_SFREQ,
        )
        eeg = np.random.randn(TINY_N_CHANS, TINY_N_TIMES).astype(np.float32) * 20
        result = clf.classify_raw(eeg_data=eeg, sample_rate=TINY_SFREQ)

        # Primary confidence must be >= any secondary confidence
        for sec in result.secondary:
            assert result.confidence >= sec["confidence"]


class TestChannelMismatch:
    def test_more_channels_truncated(self, tiny_model_path: str):
        """More channels than model expects should be truncated."""
        clf = DeepClassifier(
            model_path=tiny_model_path,
            n_chans=TINY_N_CHANS,
            n_times=TINY_N_TIMES,
            n_outputs=TINY_N_OUTPUTS,
            target_sfreq=TINY_SFREQ,
        )
        # Provide 8 channels, model expects 4
        eeg = np.random.randn(8, TINY_N_TIMES).astype(np.float32) * 20
        result = clf.classify_raw(eeg_data=eeg, sample_rate=TINY_SFREQ)
        assert result.primary in DEFAULT_LABEL_MAP.values()

    def test_fewer_channels_padded(self, tiny_model_path: str):
        """Fewer channels than model expects should be zero-padded."""
        clf = DeepClassifier(
            model_path=tiny_model_path,
            n_chans=TINY_N_CHANS,
            n_times=TINY_N_TIMES,
            n_outputs=TINY_N_OUTPUTS,
            target_sfreq=TINY_SFREQ,
        )
        # Provide 2 channels, model expects 4
        eeg = np.random.randn(2, TINY_N_TIMES).astype(np.float32) * 20
        result = clf.classify_raw(eeg_data=eeg, sample_rate=TINY_SFREQ)
        assert result.primary in DEFAULT_LABEL_MAP.values()


class TestResampling:
    def test_input_at_higher_sample_rate(self, tiny_model_path: str):
        """Input at 250Hz should be resampled to model's target_sfreq."""
        clf = DeepClassifier(
            model_path=tiny_model_path,
            n_chans=TINY_N_CHANS,
            n_times=TINY_N_TIMES,
            n_outputs=TINY_N_OUTPUTS,
            target_sfreq=TINY_SFREQ,
        )
        # 250 samples at 250Hz = 1 second; will be resampled to 64 samples at 64Hz
        eeg = np.random.randn(TINY_N_CHANS, 250).astype(np.float32) * 20
        result = clf.classify_raw(eeg_data=eeg, sample_rate=250)
        assert result.primary in DEFAULT_LABEL_MAP.values()
        assert 0.0 <= result.confidence <= 1.0

    def test_short_input_padded(self, tiny_model_path: str):
        """Input shorter than n_times should still work (zero-padded at start)."""
        clf = DeepClassifier(
            model_path=tiny_model_path,
            n_chans=TINY_N_CHANS,
            n_times=TINY_N_TIMES,
            n_outputs=TINY_N_OUTPUTS,
            target_sfreq=TINY_SFREQ,
        )
        # Only 30 samples at matching rate -> will be padded to 64
        eeg = np.random.randn(TINY_N_CHANS, 30).astype(np.float32) * 20
        result = clf.classify_raw(eeg_data=eeg, sample_rate=TINY_SFREQ)
        assert result.primary in DEFAULT_LABEL_MAP.values()


class TestGracefulErrors:
    def test_missing_model_file(self, tmp_path: Path):
        """Missing model file should not crash; is_available is False."""
        clf = DeepClassifier(
            model_path=str(tmp_path / "does_not_exist.pt"),
            n_chans=TINY_N_CHANS,
            n_times=TINY_N_TIMES,
            n_outputs=TINY_N_OUTPUTS,
            target_sfreq=TINY_SFREQ,
        )
        assert not clf.is_available

    def test_missing_model_returns_unknown(self, tmp_path: Path):
        """classify_raw with no loaded model returns 'unknown'."""
        clf = DeepClassifier(
            model_path=str(tmp_path / "does_not_exist.pt"),
            n_chans=TINY_N_CHANS,
            n_times=TINY_N_TIMES,
            n_outputs=TINY_N_OUTPUTS,
            target_sfreq=TINY_SFREQ,
        )
        eeg = np.random.randn(TINY_N_CHANS, TINY_N_TIMES).astype(np.float32) * 20
        result = clf.classify_raw(eeg_data=eeg, sample_rate=TINY_SFREQ)
        assert result.primary == "unknown"
        assert result.confidence == 0.0
