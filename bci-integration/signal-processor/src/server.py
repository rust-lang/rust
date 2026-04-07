"""FastAPI HTTP server for the BCI State Server.

Endpoints:
    GET /state  - Returns current BCIState wrapped in state_response
    GET /health - Returns server health status
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from . import config
from .brainflow_reader import BCIReader
from .models import BCIStateModel, HealthResponse, StateResponse
from .recorder import SessionRecorder
from .replayer import SessionReplayer
from .state_manager import StateManager

logger = logging.getLogger(__name__)

# Module-level references set by create_app()
_state_manager: StateManager | None = None
_reader: BCIReader | None = None
_replayer: SessionReplayer | None = None
_recorder: SessionRecorder | None = None
_start_time: float = 0.0


def create_app(
    synthetic: bool = True,
    recorder: SessionRecorder | None = None,
    replayer: SessionReplayer | None = None,
    state_manager: StateManager | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        synthetic: Whether to use BrainFlow synthetic board.
        recorder: Optional recorder to capture states during acquisition.
        replayer: Optional replayer to use instead of BrainFlow.
        state_manager: Optional pre-created StateManager (used by replayer).

    Returns:
        Configured FastAPI app instance.
    """
    global _state_manager, _reader, _replayer, _recorder, _start_time

    _state_manager = state_manager if state_manager is not None else StateManager()
    _recorder = recorder
    _replayer = replayer

    if replayer is not None:
        # Replay mode: replayer already has a reference to the state_manager
        _reader = None
    else:
        _reader = BCIReader(
            state_manager=_state_manager, synthetic=synthetic, recorder=recorder
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _start_time
        _start_time = time.time()
        logger.info("Starting BCI Signal Processor...")
        if _recorder is not None:
            _recorder.start()
        if _replayer is not None:
            _replayer.start()
        elif _reader is not None:
            _reader.start()
        yield
        logger.info("Shutting down BCI Signal Processor...")
        if _replayer is not None:
            _replayer.stop()
        elif _reader is not None:
            _reader.stop()
        if _recorder is not None:
            _recorder.stop()

    app = FastAPI(
        title="BCI State Server",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/state", response_model=StateResponse)
    def get_state() -> StateResponse:
        """Return current BCI state (state_response schema)."""
        now_ms = int(time.time() * 1000)
        state = _state_manager.get_state()

        if state is None:
            return StateResponse(
                available=False,
                timestamp_unix_ms=now_ms,
                error="No BCI data received yet",
            )

        staleness = state.get("staleness_ms", 0)

        # If data is stale (>2s), mark signal quality as 0 and state as unknown
        if staleness > config.STALE_THRESHOLD_MS:
            state["signal_quality"] = 0.0
            state["state"] = {
                "primary": "unknown",
                "confidence": 0.0,
                "secondary": [],
            }
            state["natural_language_summary"] = (
                f"User brain state: UNKNOWN - BCI data is stale "
                f"({staleness}ms old), signal unreliable."
            )

        return StateResponse(
            available=True,
            timestamp_unix_ms=now_ms,
            bci_state=BCIStateModel(**state),
        )

    @app.get("/health", response_model=HealthResponse)
    def get_health() -> HealthResponse:
        """Return server health (health_response schema)."""
        now = time.time()
        uptime = now - _start_time if _start_time > 0 else 0.0
        last_update = _state_manager.last_update_ms
        device_connected = _state_manager.device_connected

        # Determine status
        if not device_connected or last_update is None:
            status = "no_signal"
        else:
            staleness_ms = int(now * 1000) - last_update
            state = _state_manager.get_state()
            sq = state.get("signal_quality", 0) if state else 0

            if staleness_ms > config.STALE_WARNING_MS or sq < 0.3:
                status = "degraded"
            else:
                status = "ok"

        return HealthResponse(
            status=status,
            uptime_seconds=round(uptime, 1),
            device_connected=device_connected,
            session_id=_state_manager.session_id,
            last_update_unix_ms=last_update,
        )

    return app
