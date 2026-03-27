"""Tests for DSP functions.

Key test: a 10Hz sine wave should produce high alpha band power.
"""

import numpy as np
import pytest

from src.dsp import (
    assess_signal_quality,
    compute_attention,
    compute_band_powers,
    compute_cognitive_load,
    compute_relaxation,
    estimate_artifact_probability,
    sanitize_data,
)


SAMPLE_RATE = 250


def _generate_sine(freq_hz: float, duration_s: float = 1.0, amplitude: float = 20.0) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.arange(0, duration_s, 1.0 / SAMPLE_RATE)
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


class TestComputeBandPowers:
    def test_10hz_sine_has_high_alpha(self):
        """A 10Hz sine wave falls in the alpha band (8-13Hz)."""
        data = _generate_sine(10.0)
        powers = compute_band_powers(data, SAMPLE_RATE)
        # Alpha should dominate
        assert powers["alpha"] > powers["delta"]
        assert powers["alpha"] > powers["theta"]
        assert powers["alpha"] > powers["beta"]
        assert powers["alpha"] > powers["gamma"]

    def test_3hz_sine_has_high_delta(self):
        """A 3Hz sine wave falls in the delta band (1-4Hz)."""
        data = _generate_sine(3.0)
        powers = compute_band_powers(data, SAMPLE_RATE)
        assert powers["delta"] > powers["alpha"]
        assert powers["delta"] > powers["beta"]

    def test_6hz_sine_has_high_theta(self):
        """A 6Hz sine wave falls in the theta band (4-8Hz)."""
        data = _generate_sine(6.0)
        powers = compute_band_powers(data, SAMPLE_RATE)
        assert powers["theta"] > powers["delta"]
        assert powers["theta"] > powers["alpha"]
        assert powers["theta"] > powers["beta"]

    def test_20hz_sine_has_high_beta(self):
        """A 20Hz sine wave falls in the beta band (13-30Hz)."""
        data = _generate_sine(20.0)
        powers = compute_band_powers(data, SAMPLE_RATE)
        assert powers["beta"] > powers["delta"]
        assert powers["beta"] > powers["theta"]
        assert powers["beta"] > powers["alpha"]

    def test_50hz_sine_has_high_gamma(self):
        """A 50Hz sine wave falls in the gamma band (30-100Hz)."""
        data = _generate_sine(50.0)
        powers = compute_band_powers(data, SAMPLE_RATE)
        assert powers["gamma"] > powers["delta"]
        assert powers["gamma"] > powers["theta"]
        assert powers["gamma"] > powers["alpha"]
        assert powers["gamma"] > powers["beta"]

    def test_2d_multichannel(self):
        """Band powers work with 2D (channels x samples) input."""
        ch1 = _generate_sine(10.0)
        ch2 = _generate_sine(10.0, amplitude=30.0)
        data = np.stack([ch1, ch2])
        powers = compute_band_powers(data, SAMPLE_RATE)
        assert powers["alpha"] > powers["beta"]

    def test_empty_short_data(self):
        """Very short data returns zeros."""
        data = np.array([1.0, 2.0])
        powers = compute_band_powers(data, SAMPLE_RATE)
        assert all(v == 0.0 for v in powers.values())

    def test_all_nan(self):
        """All NaN channel is skipped, result is zero."""
        data = np.full(250, np.nan)
        powers = compute_band_powers(data, SAMPLE_RATE)
        assert all(v == 0.0 for v in powers.values())


class TestComputeAttention:
    def test_high_beta_high_attention(self):
        """High beta/theta ratio should give high attention."""
        powers = {"delta": 1.0, "theta": 1.0, "alpha": 1.0, "beta": 5.0, "gamma": 0.5}
        score = compute_attention(powers)
        assert score > 0.7

    def test_high_theta_low_attention(self):
        """High theta relative to beta should give low attention."""
        powers = {"delta": 1.0, "theta": 5.0, "alpha": 1.0, "beta": 0.5, "gamma": 0.5}
        score = compute_attention(powers)
        assert score < 0.3

    def test_range_0_to_1(self):
        """Score should always be in [0, 1]."""
        for beta, theta in [(0, 0), (100, 0.01), (0.01, 100)]:
            powers = {"delta": 1, "theta": theta, "alpha": 1, "beta": beta, "gamma": 1}
            score = compute_attention(powers)
            assert 0.0 <= score <= 1.0


class TestComputeRelaxation:
    def test_high_alpha_high_relaxation(self):
        """High alpha proportion should give high relaxation."""
        powers = {"delta": 0.5, "theta": 0.5, "alpha": 5.0, "beta": 0.5, "gamma": 0.5}
        score = compute_relaxation(powers)
        assert score > 0.7

    def test_low_alpha_low_relaxation(self):
        powers = {"delta": 5.0, "theta": 5.0, "alpha": 0.1, "beta": 5.0, "gamma": 5.0}
        score = compute_relaxation(powers)
        assert score < 0.3


class TestComputeCognitiveLoad:
    def test_high_theta_alpha_high_load(self):
        powers = {"delta": 0.5, "theta": 5.0, "alpha": 5.0, "beta": 0.5, "gamma": 0.5}
        score = compute_cognitive_load(powers)
        assert score > 0.7

    def test_low_theta_alpha_low_load(self):
        powers = {"delta": 5.0, "theta": 0.1, "alpha": 0.1, "beta": 5.0, "gamma": 5.0}
        score = compute_cognitive_load(powers)
        assert score < 0.3


class TestEstimateArtifactProbability:
    def test_clean_signal(self):
        data = _generate_sine(10.0, amplitude=20.0)
        prob = estimate_artifact_probability(data, threshold_uv=100.0)
        assert prob < 0.1

    def test_noisy_signal(self):
        data = np.full(250, 200.0)  # All above threshold
        prob = estimate_artifact_probability(data, threshold_uv=100.0)
        assert prob == 1.0

    def test_nan_data(self):
        data = np.full(250, np.nan)
        prob = estimate_artifact_probability(data)
        assert prob == 1.0


class TestAssessSignalQuality:
    def test_good_signal(self):
        data = _generate_sine(10.0, amplitude=20.0)
        quality = assess_signal_quality(data)
        assert quality > 0.7

    def test_flatline(self):
        data = np.zeros(250)
        quality = assess_signal_quality(data)
        assert quality < 0.2

    def test_empty(self):
        data = np.array([])
        quality = assess_signal_quality(data)
        assert quality == 0.0

    def test_nan_data(self):
        data = np.full(250, np.nan)
        quality = assess_signal_quality(data)
        assert quality == 0.0


class TestSanitizeData:
    def test_replaces_nan(self):
        data = np.array([1.0, np.nan, 3.0])
        clean = sanitize_data(data)
        assert np.all(np.isfinite(clean))
        assert clean[1] == 0.0

    def test_replaces_inf(self):
        data = np.array([1.0, np.inf, -np.inf])
        clean = sanitize_data(data)
        assert np.all(np.isfinite(clean))

    def test_no_mutation(self):
        data = np.array([1.0, np.nan, 3.0])
        _ = sanitize_data(data)
        assert np.isnan(data[1])  # Original unchanged
