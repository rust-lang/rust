"""BrainFlow data acquisition and DSP pipeline.

Reads EEG data from a BrainFlow board (synthetic or real), runs the DSP
pipeline, classifies brain state, and updates the state manager.
"""

import logging
import threading
import time
import uuid

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

from . import config
from .classifier import Classifier, HeuristicClassifier
from .dsp import (
    assess_signal_quality,
    compute_attention,
    compute_band_powers,
    compute_cognitive_load,
    compute_relaxation,
    estimate_artifact_probability,
    sanitize_data,
)
from .state_manager import StateManager

logger = logging.getLogger(__name__)


def _generate_nl_summary(
    primary_state: str,
    confidence: float,
    attention: float,
    relaxation: float,
    cognitive_load: float,
    signal_quality: float,
) -> str:
    """Generate a natural language summary for LLM context injection."""
    quality_label = "good" if signal_quality >= 0.6 else "fair" if signal_quality >= 0.3 else "poor"
    return (
        f"User brain state: {primary_state.upper()} "
        f"(confidence: {confidence:.2f}, "
        f"attention: {attention:.2f}, "
        f"relaxation: {relaxation:.2f}, "
        f"cognitive_load: {cognitive_load:.2f}, "
        f"signal quality: {quality_label})"
    )


class BCIReader:
    """Manages BrainFlow board connection and background data acquisition.

    Args:
        state_manager: Thread-safe state storage to write results to.
        synthetic: If True, use BrainFlow synthetic board (no hardware needed).
        board_id: BrainFlow board ID override. Ignored if synthetic=True.
    """

    def __init__(
        self,
        state_manager: StateManager,
        synthetic: bool = True,
        board_id: int | None = None,
        recorder: "SessionRecorder | None" = None,
        classifier: Classifier | None = None,
    ) -> None:
        self._state_manager = state_manager
        self._synthetic = synthetic
        self._recorder = recorder
        self._classifier: Classifier = classifier if classifier is not None else HeuristicClassifier()
        self._session_id = f"session-{uuid.uuid4().hex[:12]}"

        # Set up BrainFlow board
        params = BrainFlowInputParams()
        if synthetic:
            self._board_id = BoardIds.SYNTHETIC_BOARD.value
        else:
            self._board_id = board_id if board_id is not None else BoardIds.SYNTHETIC_BOARD.value

        self._board = BoardShim(self._board_id, params)
        self._eeg_channels: list[int] = []
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def device_id(self) -> str:
        return f"brainflow-board-{self._board_id}"

    def start(self) -> None:
        """Prepare and start the BrainFlow session, launch background thread."""
        logger.info("Starting BrainFlow session (board_id=%d, synthetic=%s)", self._board_id, self._synthetic)
        self._board.prepare_session()
        self._eeg_channels = BoardShim.get_eeg_channels(self._board_id)
        self._board.start_stream()
        self._state_manager.device_connected = True

        self._running = True
        self._thread = threading.Thread(target=self._acquisition_loop, daemon=True, name="bci-reader")
        self._thread.start()
        logger.info("BrainFlow streaming started. EEG channels: %s", self._eeg_channels)

    def stop(self) -> None:
        """Stop the background thread and release BrainFlow session."""
        logger.info("Stopping BrainFlow reader...")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        try:
            self._board.stop_stream()
        except Exception:
            logger.warning("Error stopping BrainFlow stream", exc_info=True)

        try:
            self._board.release_session()
        except Exception:
            logger.warning("Error releasing BrainFlow session", exc_info=True)

        self._state_manager.device_connected = False
        logger.info("BrainFlow session released.")

    def _acquisition_loop(self) -> None:
        """Background loop: reads data every WINDOW_STEP_MS, runs DSP, updates state."""
        step_seconds = config.WINDOW_STEP_MS / 1000.0
        window_samples = config.WINDOW_SIZE_SAMPLES

        while self._running:
            try:
                # Get current data from board buffer
                data = self._board.get_current_board_data(window_samples)

                if data.shape[1] < 4:
                    # Not enough samples yet
                    time.sleep(step_seconds)
                    continue

                # Extract EEG channels
                eeg_data = data[self._eeg_channels, :]

                # Sanitize: replace NaN/Inf with 0
                has_bad_values = not np.all(np.isfinite(eeg_data))
                eeg_data = sanitize_data(eeg_data)

                # DSP pipeline
                signal_quality = assess_signal_quality(eeg_data)
                artifact_prob = estimate_artifact_probability(eeg_data)

                if has_bad_values:
                    artifact_prob = 1.0
                    logger.warning("NaN/Inf detected in EEG data, setting artifact_probability=1")

                band_powers = compute_band_powers(eeg_data, config.SAMPLE_RATE)
                attention = compute_attention(band_powers)
                relaxation = compute_relaxation(band_powers)
                cog_load = compute_cognitive_load(band_powers)

                # Classify
                result = self._classifier.classify(
                    band_powers=band_powers,
                    attention=attention,
                    relaxation=relaxation,
                    cognitive_load=cog_load,
                    signal_quality=signal_quality,
                )

                # Generate NL summary
                nl_summary = _generate_nl_summary(
                    primary_state=result.primary,
                    confidence=result.confidence,
                    attention=attention,
                    relaxation=relaxation,
                    cognitive_load=cog_load,
                    signal_quality=signal_quality,
                )

                # Build BCIState
                now_ms = int(time.time() * 1000)
                bci_state = {
                    "timestamp_unix_ms": now_ms,
                    "session_id": self._session_id,
                    "device_id": self.device_id,
                    "state": {
                        "primary": result.primary,
                        "confidence": result.confidence,
                        "secondary": result.secondary,
                    },
                    "scores": {
                        "attention": round(attention, 3),
                        "relaxation": round(relaxation, 3),
                        "cognitive_load": round(cog_load, 3),
                    },
                    "band_powers": {k: round(v, 6) for k, v in band_powers.items()},
                    "signal_quality": round(signal_quality, 3),
                    "artifact_probability": round(artifact_prob, 3),
                    "staleness_ms": 0,
                    "natural_language_summary": nl_summary,
                }

                self._state_manager.update_state(bci_state)

                if self._recorder is not None:
                    self._recorder.record(bci_state)

            except Exception:
                logger.error("Error in acquisition loop", exc_info=True)

            time.sleep(step_seconds)
