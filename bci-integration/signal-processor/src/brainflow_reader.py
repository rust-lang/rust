"""BrainFlow data acquisition and DSP pipeline.

Reads EEG data from a BrainFlow board (synthetic or hardware),
runs the DSP pipeline, classifies brain state, and updates
the StateManager.
"""

import logging
import threading
import time
import uuid

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

from . import config
from .classifier import HeuristicClassifier
from .dsp import (
    assess_signal_quality,
    compute_attention,
    compute_band_powers,
    compute_cognitive_load,
    compute_relaxation,
    estimate_artifact_probability,
)
from .state_manager import StateManager

logger = logging.getLogger(__name__)


def _build_nl_summary(
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
    """Reads EEG data from BrainFlow and runs the DSP pipeline.

    Args:
        state_manager: Thread-safe state storage to update.
        synthetic: If True, use BrainFlow synthetic board. Otherwise, use Galea.
        board_id: Override board ID (for testing). Defaults based on synthetic flag.
    """

    def __init__(
        self,
        state_manager: StateManager,
        synthetic: bool = True,
        board_id: int | None = None,
    ) -> None:
        self._state_manager = state_manager
        self._synthetic = synthetic
        self._classifier = HeuristicClassifier()
        self._session_id = f"session-{uuid.uuid4().hex[:12]}"
        self._running = False
        self._thread: threading.Thread | None = None

        if board_id is not None:
            self._board_id = board_id
        elif synthetic:
            self._board_id = BoardIds.SYNTHETIC_BOARD.value
        else:
            self._board_id = BoardIds.GALEA_BOARD_V4.value

        params = BrainFlowInputParams()
        self._board = BoardShim(self._board_id, params)
        self._eeg_channels = BoardShim.get_eeg_channels(self._board_id)
        self._sample_rate = BoardShim.get_sampling_rate(self._board_id)
        self._device_id = f"{'synthetic' if synthetic else 'galea'}-{self._board_id}"

    def start(self) -> None:
        """Prepare session, start stream, launch background reader thread."""
        logger.info("Preparing BrainFlow session (board_id=%d)", self._board_id)
        self._board.prepare_session()
        self._board.start_stream()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="bci-reader")
        self._thread.start()
        logger.info("BCI reader started (session=%s, device=%s)", self._session_id, self._device_id)

    def stop(self) -> None:
        """Stop stream, release session, join background thread."""
        logger.info("Stopping BCI reader...")
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        try:
            self._board.stop_stream()
        except Exception:
            logger.warning("Error stopping stream (may already be stopped)")
        try:
            self._board.release_session()
        except Exception:
            logger.warning("Error releasing session (may already be released)")
        logger.info("BCI reader stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def device_id(self) -> str:
        return self._device_id

    def _read_loop(self) -> None:
        """Background loop: read data every WINDOW_STEP_MS, run DSP, update state."""
        step_sec = config.WINDOW_STEP_MS / 1000.0
        window_samples = config.WINDOW_SIZE_SAMPLES

        while self._running:
            try:
                time.sleep(step_sec)
                if not self._running:
                    break

                # Get current board data (up to window_samples)
                data = self._board.get_current_board_data(window_samples)
                if data.shape[1] < 4:
                    # Not enough data yet
                    continue

                # Extract EEG channels
                eeg_data = data[self._eeg_channels, :]

                # Check for NaN/Inf contamination
                has_bad_values = not np.all(np.isfinite(eeg_data))
                if has_bad_values:
                    logger.warning("NaN/Inf detected in EEG data, sanitizing")

                # DSP pipeline
                band_powers = compute_band_powers(eeg_data, self._sample_rate)
                attention = compute_attention(band_powers)
                relaxation = compute_relaxation(band_powers)
                cog_load = compute_cognitive_load(band_powers)
                artifact_prob = estimate_artifact_probability(
                    eeg_data, threshold_uv=config.ARTIFACT_AMPLITUDE_UV
                )
                sig_quality = assess_signal_quality(eeg_data)

                if has_bad_values:
                    artifact_prob = 1.0

                # Classify
                result = self._classifier.classify(
                    band_powers, attention, relaxation, cog_load
                )

                # Build NL summary
                summary = _build_nl_summary(
                    result.primary, result.confidence,
                    attention, relaxation, cog_load, sig_quality,
                )

                now_ms = int(time.time() * 1000)

                # Build BCIState
                bci_state: dict = {
                    "timestamp_unix_ms": now_ms,
                    "session_id": self._session_id,
                    "device_id": self._device_id,
                    "state": {
                        "primary": result.primary,
                        "confidence": result.confidence,
                        "secondary": result.secondary,
                    },
                    "scores": {
                        "attention": round(attention, 4),
                        "relaxation": round(relaxation, 4),
                        "cognitive_load": round(cog_load, 4),
                    },
                    "band_powers": {k: round(v, 6) for k, v in band_powers.items()},
                    "signal_quality": round(sig_quality, 4),
                    "artifact_probability": round(artifact_prob, 4),
                    "staleness_ms": 0,
                    "natural_language_summary": summary,
                }

                self._state_manager.update_state(bci_state)

            except Exception:
                logger.exception("Error in BCI read loop")
                time.sleep(1.0)
