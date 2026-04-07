"""Brain state classifiers: heuristic and ML-based.

Provides a Classifier protocol, a HeuristicClassifier (threshold rules),
and an MLClassifier (Random Forest via scikit-learn).
"""

import logging
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    primary: str
    confidence: float
    secondary: list[dict[str, object]]


@runtime_checkable
class Classifier(Protocol):
    """Protocol for brain state classifiers."""

    def classify(
        self,
        band_powers: dict[str, float],
        attention: float,
        relaxation: float,
        cognitive_load: float,
        signal_quality: float,
    ) -> ClassificationResult: ...


class HeuristicClassifier:
    """Classifies brain state using heuristic threshold rules on EEG features."""

    # Valid states matching bci_state.schema.json
    STATES = ("focused", "relaxed", "stressed", "drowsy", "meditative", "active", "unknown")

    def classify(
        self,
        band_powers: dict[str, float],
        attention: float,
        relaxation: float,
        cognitive_load: float,
        signal_quality: float,
    ) -> ClassificationResult:
        """Classify brain state from features.

        Args:
            band_powers: Dict with delta/theta/alpha/beta/gamma power values.
            attention: Attention score 0-1.
            relaxation: Relaxation score 0-1.
            cognitive_load: Cognitive load score 0-1.
            signal_quality: Signal quality 0-1.

        Returns:
            ClassificationResult with primary state, confidence, and secondary states.
        """
        # If signal quality is too low, return unknown
        if signal_quality < 0.2:
            return ClassificationResult(
                primary="unknown",
                confidence=0.0,
                secondary=[],
            )

        # Compute candidate scores for each state
        candidates: dict[str, float] = {}

        total = sum(band_powers.values())
        if total <= 0:
            return ClassificationResult(primary="unknown", confidence=0.0, secondary=[])

        delta_ratio = band_powers.get("delta", 0.0) / total
        theta_ratio = band_powers.get("theta", 0.0) / total
        alpha_ratio = band_powers.get("alpha", 0.0) / total
        beta_ratio = band_powers.get("beta", 0.0) / total
        gamma_ratio = band_powers.get("gamma", 0.0) / total

        # Focused: high beta, high attention, low alpha
        candidates["focused"] = (
            0.4 * attention
            + 0.3 * min(beta_ratio / 0.3, 1.0)
            + 0.2 * (1.0 - relaxation)
            + 0.1 * (1.0 - min(delta_ratio / 0.3, 1.0))
        )

        # Relaxed: high alpha, high relaxation, low beta
        candidates["relaxed"] = (
            0.4 * relaxation
            + 0.3 * min(alpha_ratio / 0.3, 1.0)
            + 0.2 * (1.0 - attention)
            + 0.1 * (1.0 - min(beta_ratio / 0.3, 1.0))
        )

        # Stressed: high beta + high gamma, high cognitive load
        candidates["stressed"] = (
            0.3 * cognitive_load
            + 0.3 * min((beta_ratio + gamma_ratio) / 0.4, 1.0)
            + 0.2 * (1.0 - relaxation)
            + 0.2 * attention
        )

        # Drowsy: high theta + high delta, low beta
        candidates["drowsy"] = (
            0.4 * min(theta_ratio / 0.3, 1.0)
            + 0.3 * min(delta_ratio / 0.3, 1.0)
            + 0.2 * (1.0 - attention)
            + 0.1 * (1.0 - min(beta_ratio / 0.3, 1.0))
        )

        # Meditative: high alpha + high theta, low beta, moderate relaxation
        candidates["meditative"] = (
            0.3 * min(alpha_ratio / 0.3, 1.0)
            + 0.3 * min(theta_ratio / 0.3, 1.0)
            + 0.2 * relaxation
            + 0.2 * (1.0 - min(beta_ratio / 0.3, 1.0))
        )

        # Active: high gamma + high beta, moderate attention
        candidates["active"] = (
            0.3 * min(gamma_ratio / 0.2, 1.0)
            + 0.3 * min(beta_ratio / 0.3, 1.0)
            + 0.2 * attention
            + 0.2 * cognitive_load
        )

        # Clamp all candidate scores to [0, 1]
        for state in candidates:
            candidates[state] = max(0.0, min(1.0, candidates[state]))

        # Sort by score descending
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        primary_state = ranked[0][0]
        primary_score = ranked[0][1]

        # Confidence: how much the primary stands out from the second
        if len(ranked) > 1:
            gap = primary_score - ranked[1][1]
            # Confidence is based on both absolute score and gap
            confidence = min(1.0, 0.5 * primary_score + 0.5 * (gap / max(primary_score, 0.01)))
        else:
            confidence = primary_score

        # Scale confidence by signal quality
        confidence *= signal_quality

        secondary = [
            {"state": state, "confidence": round(score * signal_quality, 3)}
            for state, score in ranked[1:]
            if score > 0.1
        ]

        return ClassificationResult(
            primary=primary_state,
            confidence=round(confidence, 3),
            secondary=secondary,
        )


class MLClassifier:
    """Classifies brain state using a trained scikit-learn model.

    Falls back to HeuristicClassifier if the model file doesn't exist.

    Args:
        model_path: Path to a joblib-serialized sklearn model.
    """

    # Feature vector order -- must match train_model.py
    FEATURE_NAMES = (
        "delta", "theta", "alpha", "beta", "gamma",
        "attention", "relaxation", "cognitive_load", "signal_quality",
    )

    # Label order used during training -- must match train_model.py
    STATES = ("focused", "relaxed", "stressed", "drowsy", "meditative", "active")

    def __init__(self, model_path: str) -> None:
        self._fallback: HeuristicClassifier | None = None
        self._model = None
        self._classes: list[str] = []

        if not os.path.isfile(model_path):
            logger.warning(
                "ML model file not found at %s -- falling back to HeuristicClassifier",
                model_path,
            )
            self._fallback = HeuristicClassifier()
            return

        try:
            import joblib

            self._model = joblib.load(model_path)
            self._classes = list(self._model.classes_)
            logger.info("Loaded ML classifier from %s (classes: %s)", model_path, self._classes)
        except Exception:
            logger.exception("Failed to load ML model from %s -- falling back to heuristic", model_path)
            self._fallback = HeuristicClassifier()

    def classify(
        self,
        band_powers: dict[str, float],
        attention: float,
        relaxation: float,
        cognitive_load: float,
        signal_quality: float,
    ) -> ClassificationResult:
        if self._fallback is not None:
            return self._fallback.classify(
                band_powers, attention, relaxation, cognitive_load, signal_quality,
            )

        # Low signal quality guard (same as heuristic)
        if signal_quality < 0.2:
            return ClassificationResult(primary="unknown", confidence=0.0, secondary=[])

        import numpy as np

        feature_vector = np.array([[
            band_powers.get("delta", 0.0),
            band_powers.get("theta", 0.0),
            band_powers.get("alpha", 0.0),
            band_powers.get("beta", 0.0),
            band_powers.get("gamma", 0.0),
            attention,
            relaxation,
            cognitive_load,
            signal_quality,
        ]])

        probas = self._model.predict_proba(feature_vector)[0]

        # Pair classes with probabilities, sort descending
        ranked = sorted(zip(self._classes, probas), key=lambda x: x[1], reverse=True)

        primary_state = ranked[0][0]
        primary_prob = float(ranked[0][1])

        # Scale confidence by signal quality
        confidence = round(primary_prob * signal_quality, 3)

        secondary = [
            {"state": state, "confidence": round(float(prob) * signal_quality, 3)}
            for state, prob in ranked[1:]
            if prob > 0.05
        ]

        return ClassificationResult(
            primary=primary_state,
            confidence=confidence,
            secondary=secondary,
        )


def create_classifier(model_path: str | None = None) -> Classifier:
    """Factory: create the appropriate classifier.

    Args:
        model_path: Path to a joblib ML model file. If None, uses heuristic.

    Returns:
        A Classifier instance.
    """
    if model_path is not None:
        return MLClassifier(model_path)
    return HeuristicClassifier()
