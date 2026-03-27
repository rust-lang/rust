"""Heuristic brain state classifier.

Classifies mental state from band powers and derived scores using
threshold-based rules. No ML training needed.
"""

from dataclasses import dataclass


@dataclass
class ClassificationResult:
    primary: str
    confidence: float
    secondary: list[dict]  # [{"state": str, "confidence": float}, ...]


class HeuristicClassifier:
    """Classifies brain state from EEG features using heuristic rules.

    Rules (evaluated in priority order):
    - High alpha + high relaxation -> relaxed
    - High beta + high attention -> focused
    - High theta + low attention -> drowsy
    - High beta + high cognitive_load + low relaxation -> stressed
    - High alpha + high theta + high relaxation -> meditative
    - High beta + high gamma -> active
    - Otherwise -> unknown
    """

    def classify(
        self,
        band_powers: dict[str, float],
        attention: float,
        relaxation: float,
        cognitive_load: float,
    ) -> ClassificationResult:
        """Classify brain state from features.

        Args:
            band_powers: Dict with delta/theta/alpha/beta/gamma power values.
            attention: Attention score 0-1.
            relaxation: Relaxation score 0-1.
            cognitive_load: Cognitive load score 0-1.

        Returns:
            ClassificationResult with primary state, confidence, and secondary states.
        """
        total = sum(band_powers.values())
        if total <= 0:
            return ClassificationResult(
                primary="unknown", confidence=0.0, secondary=[]
            )

        # Normalize band powers to relative proportions
        rel = {k: v / total for k, v in band_powers.items()}

        # Score each state
        scores: dict[str, float] = {}

        # Relaxed: high alpha dominance + high relaxation score
        scores["relaxed"] = (rel.get("alpha", 0) * 2.0 + relaxation) / 3.0

        # Focused: high beta + high attention
        scores["focused"] = (rel.get("beta", 0) * 2.0 + attention) / 3.0

        # Drowsy: high theta + low attention
        scores["drowsy"] = (rel.get("theta", 0) * 2.0 + (1.0 - attention)) / 3.0

        # Stressed: high beta + high cognitive load + low relaxation
        scores["stressed"] = (
            rel.get("beta", 0) + cognitive_load + (1.0 - relaxation)
        ) / 3.0

        # Meditative: high alpha + moderate theta + high relaxation
        scores["meditative"] = (
            rel.get("alpha", 0) + rel.get("theta", 0) * 0.5 + relaxation
        ) / 2.5

        # Active: high beta + high gamma
        scores["active"] = (
            rel.get("beta", 0) + rel.get("gamma", 0) * 2.0
        ) / 3.0

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary_state = ranked[0][0]
        primary_confidence = float(min(ranked[0][1], 1.0))

        # Secondary: all other states with non-trivial scores
        secondary = [
            {"state": state, "confidence": round(min(score, 1.0), 4)}
            for state, score in ranked[1:]
            if score > 0.1
        ]

        return ClassificationResult(
            primary=primary_state,
            confidence=round(primary_confidence, 4),
            secondary=secondary,
        )
