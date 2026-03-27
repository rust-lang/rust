"""Pure DSP functions for EEG signal processing.

All functions are stateless and operate on numpy arrays.
"""

import numpy as np
from scipy.signal import welch

from . import config

# Frequency band definitions (Hz)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Compute average power in a frequency band."""
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(psd[mask]))


def compute_band_powers(data: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> dict[str, float]:
    """Compute power spectral density per frequency band using Welch's method.

    Args:
        data: 1D or 2D array of EEG samples. If 2D, shape is (channels, samples)
              and results are averaged across channels.
        sample_rate: Sampling rate in Hz.

    Returns:
        Dict with keys: delta, theta, alpha, beta, gamma. Values are power (uV^2/Hz).
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_samples = data.shape[1]
    # nperseg must be <= n_samples; use min of 1 second or available data
    nperseg = min(n_samples, sample_rate)
    if nperseg < 4:
        return {band: 0.0 for band in BANDS}

    all_powers = {band: [] for band in BANDS}

    for ch in range(data.shape[0]):
        channel_data = data[ch]
        # Skip channels with NaN/Inf
        if not np.all(np.isfinite(channel_data)):
            continue

        freqs, psd = welch(channel_data, fs=sample_rate, nperseg=nperseg)

        for band, (low, high) in BANDS.items():
            all_powers[band].append(_band_power(freqs, psd, low, high))

    result = {}
    for band in BANDS:
        if all_powers[band]:
            result[band] = float(np.mean(all_powers[band]))
        else:
            result[band] = 0.0

    return result


def compute_attention(band_powers: dict[str, float]) -> float:
    """Compute attention score (0-1) from beta/theta ratio.

    Higher beta relative to theta indicates greater attentional engagement.
    """
    theta = band_powers.get("theta", 0.0)
    beta = band_powers.get("beta", 0.0)
    if theta <= 0:
        return 1.0 if beta > 0 else 0.5
    ratio = beta / theta
    # Normalize: ratio ~0.5-4.0 maps to 0-1
    score = (ratio - 0.5) / 3.5
    return float(np.clip(score, 0.0, 1.0))


def compute_relaxation(band_powers: dict[str, float]) -> float:
    """Compute relaxation score (0-1) from alpha power.

    Higher alpha relative to total power indicates relaxation.
    """
    alpha = band_powers.get("alpha", 0.0)
    total = sum(band_powers.values())
    if total <= 0:
        return 0.0
    ratio = alpha / total
    # Alpha typically 10-40% of total when relaxed; map 0.05-0.4 -> 0-1
    score = (ratio - 0.05) / 0.35
    return float(np.clip(score, 0.0, 1.0))


def compute_cognitive_load(band_powers: dict[str, float]) -> float:
    """Compute cognitive load (0-1) from theta+alpha power.

    Higher frontal theta+alpha indicates greater cognitive load.
    """
    theta = band_powers.get("theta", 0.0)
    alpha = band_powers.get("alpha", 0.0)
    total = sum(band_powers.values())
    if total <= 0:
        return 0.0
    ratio = (theta + alpha) / total
    # Map 0.2-0.7 -> 0-1
    score = (ratio - 0.2) / 0.5
    return float(np.clip(score, 0.0, 1.0))


def estimate_artifact_probability(data: np.ndarray, threshold_uv: float = config.ARTIFACT_AMPLITUDE_UV) -> float:
    """Estimate probability that the window contains artifacts.

    Uses amplitude threshold: samples exceeding threshold_uv are likely artifacts
    (eye blinks, jaw clenches, movement).

    Args:
        data: 1D or 2D array of EEG samples in microvolts.
        threshold_uv: Amplitude threshold in microvolts.

    Returns:
        Float 0-1 representing artifact probability.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if not np.all(np.isfinite(data)):
        return 1.0

    total_samples = data.size
    if total_samples == 0:
        return 1.0

    artifact_samples = np.sum(np.abs(data) > threshold_uv)
    probability = float(artifact_samples / total_samples)
    return float(np.clip(probability, 0.0, 1.0))


def assess_signal_quality(data: np.ndarray) -> float:
    """Assess overall signal quality (0-1).

    Checks for:
    - NaN/Inf values (bad)
    - Flat-line (std near zero, bad)
    - Excessive amplitude (bad)
    - Reasonable variance (good)

    Args:
        data: 1D or 2D array of EEG samples.

    Returns:
        Float 0-1 where 1.0 is excellent quality.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.size == 0:
        return 0.0

    scores = []
    for ch in range(data.shape[0]):
        channel = data[ch]

        # Check for NaN/Inf
        finite_mask = np.isfinite(channel)
        finite_ratio = np.mean(finite_mask)
        if finite_ratio < 0.5:
            scores.append(0.0)
            continue

        clean = channel[finite_mask]
        if len(clean) < 4:
            scores.append(0.0)
            continue

        # Flat-line check: std should be > 1 uV for real EEG
        std = np.std(clean)
        if std < 0.1:
            scores.append(0.1)
            continue

        # Amplitude check: most samples should be within reasonable range
        in_range = np.mean(np.abs(clean) < 150.0)

        # Combine: finite ratio * amplitude ratio
        ch_quality = finite_ratio * in_range
        # Bonus for having reasonable variance (5-50 uV std is typical for EEG)
        if 1.0 < std < 100.0:
            ch_quality *= 1.0
        else:
            ch_quality *= 0.5

        scores.append(float(np.clip(ch_quality, 0.0, 1.0)))

    if not scores:
        return 0.0

    return float(np.mean(scores))


def sanitize_data(data: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0.

    Args:
        data: numpy array, potentially containing NaN or Inf.

    Returns:
        Cleaned copy with NaN/Inf replaced by 0.
    """
    cleaned = np.copy(data)
    bad_mask = ~np.isfinite(cleaned)
    if np.any(bad_mask):
        cleaned[bad_mask] = 0.0
    return cleaned
