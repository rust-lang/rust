"""Heuristic brain state classifier.

Classifies brain state from band powers and derived scores using threshold rules.
No ML -- transparent, deterministic logic suitable for a prototype.
"""

from dataclasses import dataclass


@dataclass
class ClassificationResult:
    primary: str
    confidence: float
    secondary: list[dict[str, object]]


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
