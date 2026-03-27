"""Pure DSP functions for EEG signal processing.

All functions are stateless and operate on numpy arrays.
"""

import numpy as np
from scipy.signal import welch

# Frequency band definitions (Hz)
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def _sanitize(data: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0."""
    result = np.copy(data)
    mask = ~np.isfinite(result)
    result[mask] = 0.0
    return result


def compute_band_powers(data: np.ndarray, sample_rate: int) -> dict[str, float]:
    """Compute power spectral density per frequency band using Welch's method.

    Args:
        data: EEG data array of shape (n_channels, n_samples) or (n_samples,).
        sample_rate: Sampling rate in Hz.

    Returns:
        Dict with keys delta, theta, alpha, beta, gamma and float power values.
    """
    data = _sanitize(data)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_samples = data.shape[1]
    # nperseg must be <= n_samples; use min of n_samples and sample_rate
    nperseg = min(n_samples, sample_rate)
    if nperseg < 4:
        return {band: 0.0 for band in BANDS}

    band_powers: dict[str, list[float]] = {band: [] for band in BANDS}

    for ch in range(data.shape[0]):
        freqs, psd = welch(data[ch], fs=sample_rate, nperseg=nperseg)
        for band, (low, high) in BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            if mask.any():
                power = float(np.trapz(psd[mask], freqs[mask]))
                band_powers[band].append(max(power, 0.0))
            else:
                band_powers[band].append(0.0)

    return {band: float(np.mean(vals)) if vals else 0.0 for band, vals in band_powers.items()}


def compute_attention(band_powers: dict[str, float]) -> float:
    """Compute attention score (0-1) from beta/theta ratio.

    Higher beta relative to theta indicates focused attention.
    """
    theta = band_powers.get("theta", 0.0)
    beta = band_powers.get("beta", 0.0)
    if theta <= 0:
        return 1.0 if beta > 0 else 0.5
    ratio = beta / theta
    # Normalize: ratio of ~2 -> 1.0, ratio of ~0.5 -> 0.0
    score = (ratio - 0.5) / 1.5
    return float(np.clip(score, 0.0, 1.0))


def compute_relaxation(band_powers: dict[str, float]) -> float:
    """Compute relaxation score (0-1) from alpha power dominance.

    Higher alpha relative to total power indicates relaxation.
    """
    alpha = band_powers.get("alpha", 0.0)
    total = sum(band_powers.values())
    if total <= 0:
        return 0.0
    ratio = alpha / total
    # Alpha typically 10-20% of total when relaxed; normalize so 0.3 -> 1.0
    score = ratio / 0.3
    return float(np.clip(score, 0.0, 1.0))


def compute_cognitive_load(band_powers: dict[str, float]) -> float:
    """Compute cognitive load (0-1) from theta+alpha power.

    Higher frontal theta+alpha indicates higher cognitive load.
    """
    theta = band_powers.get("theta", 0.0)
    alpha = band_powers.get("alpha", 0.0)
    total = sum(band_powers.values())
    if total <= 0:
        return 0.0
    ratio = (theta + alpha) / total
    # theta+alpha is typically 20-50% of total; normalize so 0.5 -> 1.0
    score = ratio / 0.5
    return float(np.clip(score, 0.0, 1.0))


def estimate_artifact_probability(data: np.ndarray, threshold_uv: float = 100.0) -> float:
    """Estimate probability that the window contains artifacts.

    Uses amplitude threshold: samples exceeding threshold_uv are likely artifacts
    (eye blinks, jaw clenches, movement).

    Args:
        data: EEG data array of shape (n_channels, n_samples) or (n_samples,).
        threshold_uv: Amplitude threshold in microvolts.

    Returns:
        Float 0-1 representing artifact probability.
    """
    data = _sanitize(data)
    if data.size == 0:
        return 1.0

    exceeding = np.abs(data) > threshold_uv
    fraction = float(np.mean(exceeding))
    # Scale: if >10% of samples exceed threshold, artifact_probability = 1.0
    score = fraction / 0.1
    return float(np.clip(score, 0.0, 1.0))


def assess_signal_quality(data: np.ndarray) -> float:
    """Assess overall signal quality (0-1).

    Checks for:
    - Flatline (std dev near zero -> bad)
    - Excessive amplitude (likely artifact -> bad)
    - NaN/Inf presence (bad)

    Args:
        data: EEG data of shape (n_channels, n_samples) or (n_samples,).

    Returns:
        Float 0-1 where 1.0 is excellent quality.
    """
    raw = np.copy(data)
    if raw.size == 0:
        return 0.0

    # Check for NaN/Inf
    nan_inf_fraction = float(np.mean(~np.isfinite(raw)))
    if nan_inf_fraction > 0.5:
        return 0.0

    clean = _sanitize(raw)
    if clean.ndim == 1:
        clean = clean.reshape(1, -1)

    quality_per_channel = []
    for ch in range(clean.shape[0]):
        ch_data = clean[ch]
        std = float(np.std(ch_data))
        amp_max = float(np.max(np.abs(ch_data)))

        # Flatline check: std < 0.1 uV is suspicious
        if std < 0.1:
            quality_per_channel.append(0.1)
            continue

        # Amplitude check: penalize if max > 100 uV
        amp_penalty = min(amp_max / 200.0, 1.0)

        # NaN penalty
        nan_penalty = nan_inf_fraction

        quality = 1.0 - (amp_penalty * 0.5) - (nan_penalty * 0.5)
        quality_per_channel.append(max(quality, 0.0))

    if not quality_per_channel:
        return 0.0

    return float(np.clip(np.mean(quality_per_channel), 0.0, 1.0))
