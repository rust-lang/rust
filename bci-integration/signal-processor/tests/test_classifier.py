"""Tests for the HeuristicClassifier."""

import pytest

from src.classifier import HeuristicClassifier


@pytest.fixture
def classifier():
    return HeuristicClassifier()


class TestHeuristicClassifier:
    def test_focused_state(self, classifier):
        """High beta + high attention -> focused."""
        band_powers = {"delta": 0.5, "theta": 0.5, "alpha": 0.5, "beta": 5.0, "gamma": 1.0}
        result = classifier.classify(
            band_powers=band_powers,
            attention=0.9,
            relaxation=0.1,
            cognitive_load=0.5,
            signal_quality=0.9,
        )
        assert result.primary == "focused"
        assert result.confidence > 0.3

    def test_relaxed_state(self, classifier):
        """High alpha + high relaxation -> relaxed."""
        band_powers = {"delta": 0.5, "theta": 0.5, "alpha": 5.0, "beta": 0.3, "gamma": 0.2}
        result = classifier.classify(
            band_powers=band_powers,
            attention=0.1,
            relaxation=0.9,
            cognitive_load=0.3,
            signal_quality=0.9,
        )
        assert result.primary == "relaxed"
        assert result.confidence > 0.3

    def test_drowsy_state(self, classifier):
        """High theta + high delta -> drowsy."""
        band_powers = {"delta": 4.0, "theta": 5.0, "alpha": 0.5, "beta": 0.3, "gamma": 0.2}
        result = classifier.classify(
            band_powers=band_powers,
            attention=0.1,
            relaxation=0.2,
            cognitive_load=0.3,
            signal_quality=0.9,
        )
        assert result.primary == "drowsy"
        assert result.confidence > 0.2

    def test_unknown_on_low_signal_quality(self, classifier):
        """Low signal quality -> unknown state."""
        band_powers = {"delta": 1.0, "theta": 1.0, "alpha": 1.0, "beta": 1.0, "gamma": 1.0}
        result = classifier.classify(
            band_powers=band_powers,
            attention=0.5,
            relaxation=0.5,
            cognitive_load=0.5,
            signal_quality=0.1,
        )
        assert result.primary == "unknown"
        assert result.confidence == 0.0

    def test_unknown_on_zero_power(self, classifier):
        """All-zero band powers -> unknown."""
        band_powers = {"delta": 0.0, "theta": 0.0, "alpha": 0.0, "beta": 0.0, "gamma": 0.0}
        result = classifier.classify(
            band_powers=band_powers,
            attention=0.0,
            relaxation=0.0,
            cognitive_load=0.0,
            signal_quality=0.8,
        )
        assert result.primary == "unknown"

    def test_secondary_states_present(self, classifier):
        """Result should include secondary states."""
        band_powers = {"delta": 1.0, "theta": 1.0, "alpha": 3.0, "beta": 1.0, "gamma": 0.5}
        result = classifier.classify(
            band_powers=band_powers,
            attention=0.4,
            relaxation=0.7,
            cognitive_load=0.5,
            signal_quality=0.9,
        )
        assert len(result.secondary) > 0
        for s in result.secondary:
            assert "state" in s
            assert "confidence" in s
            assert 0.0 <= s["confidence"] <= 1.0

    def test_confidence_scaled_by_signal_quality(self, classifier):
        """Higher signal quality should give higher confidence for same features."""
        band_powers = {"delta": 0.5, "theta": 0.5, "alpha": 0.5, "beta": 5.0, "gamma": 1.0}
        kwargs = dict(band_powers=band_powers, attention=0.9, relaxation=0.1, cognitive_load=0.5)

        result_high = classifier.classify(**kwargs, signal_quality=1.0)
        result_low = classifier.classify(**kwargs, signal_quality=0.5)
        assert result_high.confidence >= result_low.confidence

    def test_primary_is_valid_state(self, classifier):
        """Primary state must be one of the valid enum values."""
        band_powers = {"delta": 2.0, "theta": 3.0, "alpha": 2.0, "beta": 1.0, "gamma": 0.5}
        result = classifier.classify(
            band_powers=band_powers,
            attention=0.3,
            relaxation=0.4,
            cognitive_load=0.6,
            signal_quality=0.8,
        )
        assert result.primary in HeuristicClassifier.STATES

    def test_confidence_in_range(self, classifier):
        """Confidence should always be in [0, 1]."""
        import random
        random.seed(42)
        for _ in range(50):
            bp = {b: random.uniform(0, 10) for b in ("delta", "theta", "alpha", "beta", "gamma")}
            result = classifier.classify(
                band_powers=bp,
                attention=random.random(),
                relaxation=random.random(),
                cognitive_load=random.random(),
                signal_quality=random.random(),
            )
            assert 0.0 <= result.confidence <= 1.0, f"Confidence {result.confidence} out of range"
