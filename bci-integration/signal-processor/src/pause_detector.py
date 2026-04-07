"""Brain-triggered pause detection system.

Detects three types of pause triggers:
1. Jaw clench (deliberate pause) - double clench pattern in high-gamma
2. Drowsiness (automatic pause) - sustained low attention
3. Headset removal - sustained zero signal quality
"""

import time
from collections import deque
from dataclasses import dataclass, asdict

import numpy as np

from . import config
from .dsp import compute_high_gamma_power


@dataclass
class PauseEvent:
    """Represents a detected pause trigger."""

    pause_type: str  # "deliberate" or "automatic"
    trigger: str  # "jaw_clench", "drowsiness", "headset_removed"
    confidence: float  # 0-1
    timestamp_unix_ms: int
    recommended_action: str  # "pause", "slow_down", "stop"

    def to_dict(self) -> dict:
        return asdict(self)


class PauseDetector:
    """Detects brain-triggered pause conditions from EEG signals.

    Monitors three channels:
    - High-gamma power for jaw clench detection (deliberate pause)
    - Attention scores for drowsiness detection (automatic pause)
    - Signal quality for headset removal detection (automatic pause)
    """

    def __init__(self) -> None:
        # Jaw clench tracking
        self._clench_times: deque[float] = deque(maxlen=10)
        self._gamma_baseline: float = 0.0
        self._gamma_baseline_count: int = 0
        self._in_clench: bool = False
        self._clench_start_ms: int = 0

        # Drowsiness tracking
        self._attention_history: deque[float] = deque(
            maxlen=config.DROWSINESS_WINDOW_SIZE
        )

        # Headset removal tracking
        self._quality_history: deque[float] = deque(
            maxlen=config.HEADSET_REMOVED_WINDOW_SIZE
        )

        # Pause state
        self._paused: bool = False
        self._pause_reason: str | None = None
        self._pause_since_ms: int | None = None

        # Cooldown: don't re-trigger within 3 seconds of a resume
        self._last_resume_ms: int = 0
        self._cooldown_ms: int = 3000

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def pause_reason(self) -> str | None:
        return self._pause_reason

    @property
    def pause_since_ms(self) -> int | None:
        return self._pause_since_ms

    def detect_clench(self, eeg_data: np.ndarray, sample_rate: int) -> bool:
        """Detect a jaw clench event from high-gamma power.

        A clench is detected when high-gamma power exceeds
        CLENCH_THRESHOLD_FACTOR * baseline for at least CLENCH_MIN_DURATION_MS.

        Args:
            eeg_data: EEG data array (channels x samples or 1D).
            sample_rate: Sampling rate in Hz.

        Returns:
            True if a completed clench event was detected this call.
        """
        gamma_power = compute_high_gamma_power(eeg_data, sample_rate)
        now_ms = int(time.time() * 1000)

        # Update baseline with exponential moving average
        if self._gamma_baseline_count == 0:
            self._gamma_baseline = gamma_power
            self._gamma_baseline_count = 1
            return False

        # EMA with alpha = 0.05 (slow adaptation)
        alpha = 0.05
        self._gamma_baseline = (
            alpha * gamma_power + (1 - alpha) * self._gamma_baseline
        )
        self._gamma_baseline_count += 1

        # Need some baseline samples before detecting
        if self._gamma_baseline_count < 4:
            return False

        threshold = self._gamma_baseline * config.CLENCH_THRESHOLD_FACTOR

        if gamma_power > threshold:
            if not self._in_clench:
                self._in_clench = True
                self._clench_start_ms = now_ms
            else:
                # Check if clench has lasted long enough
                duration_ms = now_ms - self._clench_start_ms
                if duration_ms >= config.CLENCH_MIN_DURATION_MS:
                    self._in_clench = False
                    self._clench_times.append(now_ms / 1000.0)
                    return True
        else:
            self._in_clench = False

        return False

    def check_drowsiness(self, band_powers: dict, attention: float) -> bool:
        """Check for drowsiness based on sustained low attention.

        Args:
            band_powers: Dict of frequency band powers.
            attention: Current attention score (0-1).

        Returns:
            True if drowsiness is detected (sustained low attention).
        """
        self._attention_history.append(attention)

        if len(self._attention_history) < config.DROWSINESS_WINDOW_SIZE:
            return False

        mean_attention = sum(self._attention_history) / len(self._attention_history)
        return mean_attention < config.DROWSINESS_ATTENTION_THRESHOLD

    def check_headset_removed(self, signal_quality: float) -> bool:
        """Check if the headset has been removed.

        Args:
            signal_quality: Current signal quality score (0-1).

        Returns:
            True if signal quality has been near zero for the required window.
        """
        self._quality_history.append(signal_quality)

        if len(self._quality_history) < config.HEADSET_REMOVED_WINDOW_SIZE:
            return False

        return all(
            q < config.HEADSET_REMOVED_QUALITY_THRESHOLD
            for q in self._quality_history
        )

    def update(
        self,
        eeg_data: np.ndarray,
        band_powers: dict,
        attention: float,
        signal_quality: float,
        sample_rate: int,
    ) -> PauseEvent | None:
        """Check all pause triggers.

        Returns a PauseEvent if a new pause was triggered, None otherwise.
        Does not trigger if already paused or within cooldown period.

        Args:
            eeg_data: EEG data array.
            band_powers: Dict of frequency band powers.
            attention: Current attention score (0-1).
            signal_quality: Current signal quality (0-1).
            sample_rate: Sampling rate in Hz.

        Returns:
            PauseEvent if triggered, None otherwise.
        """
        now_ms = int(time.time() * 1000)

        # Don't trigger if already paused
        if self._paused:
            return None

        # Cooldown check
        if now_ms - self._last_resume_ms < self._cooldown_ms:
            # Still update histories even during cooldown
            self.detect_clench(eeg_data, sample_rate)
            self.check_drowsiness(band_powers, attention)
            self.check_headset_removed(signal_quality)
            return None

        # Check jaw clench (deliberate pause: 2 clenches within window)
        clench_detected = self.detect_clench(eeg_data, sample_rate)
        if clench_detected:
            # Check for double clench pattern
            now_s = now_ms / 1000.0
            recent_clenches = [
                t
                for t in self._clench_times
                if (now_s - t) <= config.CLENCH_WINDOW_S
            ]
            if len(recent_clenches) >= 2:
                self._paused = True
                self._pause_reason = "jaw_clench"
                self._pause_since_ms = now_ms
                self._clench_times.clear()
                return PauseEvent(
                    pause_type="deliberate",
                    trigger="jaw_clench",
                    confidence=0.85,
                    timestamp_unix_ms=now_ms,
                    recommended_action="pause",
                )

        # Check drowsiness (automatic pause)
        if self.check_drowsiness(band_powers, attention):
            self._paused = True
            self._pause_reason = "drowsiness"
            self._pause_since_ms = now_ms
            return PauseEvent(
                pause_type="automatic",
                trigger="drowsiness",
                confidence=0.7,
                timestamp_unix_ms=now_ms,
                recommended_action="slow_down",
            )

        # Check headset removal (automatic pause)
        if self.check_headset_removed(signal_quality):
            self._paused = True
            self._pause_reason = "headset_removed"
            self._pause_since_ms = now_ms
            return PauseEvent(
                pause_type="automatic",
                trigger="headset_removed",
                confidence=0.95,
                timestamp_unix_ms=now_ms,
                recommended_action="stop",
            )

        return None

    def check_resume(self, attention: float, signal_quality: float) -> bool:
        """Check if conditions are met to resume from a paused state.

        Resume conditions:
        - jaw_clench: single clench detected, OR attention > 0.4
        - drowsiness: attention returns above threshold for a reading
        - headset_removed: signal quality returns above threshold

        Args:
            attention: Current attention score (0-1).
            signal_quality: Current signal quality (0-1).

        Returns:
            True if resumed, False otherwise.
        """
        if not self._paused:
            return False

        now_ms = int(time.time() * 1000)
        resumed = False

        if self._pause_reason == "jaw_clench":
            # Resume if attention is good (user is alert and wants to continue)
            if attention > 0.4:
                resumed = True

        elif self._pause_reason == "drowsiness":
            # Resume if attention returns above threshold
            if attention > config.DROWSINESS_ATTENTION_THRESHOLD * 2:
                resumed = True

        elif self._pause_reason == "headset_removed":
            # Resume if signal quality returns
            if signal_quality > 0.3:
                resumed = True

        if resumed:
            self._paused = False
            self._pause_reason = None
            self._pause_since_ms = None
            self._last_resume_ms = now_ms
            # Clear histories to avoid immediate re-trigger
            self._attention_history.clear()
            self._quality_history.clear()
            return True

        return False
