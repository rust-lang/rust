"""FastAPI HTTP server for the BCI State Server.

Endpoints:
    GET  /state   - Returns current BCIState wrapped in state_response
    GET  /health  - Returns server health status
    WS   /ws      - WebSocket endpoint for real-time BCI state streaming
    GET  /ws/info - WebSocket connection info for monitoring
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from . import config
from .brainflow_reader import BCIReader
from .classifier import Classifier
from .models import (
    BCIStateModel,
    HealthResponse,
    StateResponse,
    WebhookInfo,
    WebhookRegistration,
)
from .recorder import SessionRecorder
from .replayer import SessionReplayer
from .state_manager import StateManager
from .webhook_manager import WebhookManager
from .ws_manager import WebSocketManager

logger = logging.getLogger(__name__)

# Module-level references set by create_app()
_state_manager: StateManager | None = None
_reader: BCIReader | None = None
_replayer: SessionReplayer | None = None
_recorder: SessionRecorder | None = None
_webhook_manager: WebhookManager | None = None
_ws_manager: WebSocketManager | None = None
_start_time: float = 0.0


def create_app(
    synthetic: bool = True,
    recorder: SessionRecorder | None = None,
    replayer: SessionReplayer | None = None,
    state_manager: StateManager | None = None,
    classifier: Classifier | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        synthetic: Whether to use BrainFlow synthetic board.
        recorder: Optional recorder to capture states during acquisition.
        replayer: Optional replayer to use instead of BrainFlow.
        state_manager: Optional pre-created StateManager (used by replayer).
        classifier: Optional classifier override. Defaults to HeuristicClassifier.

    Returns:
        Configured FastAPI app instance.
    """
    global _state_manager, _reader, _replayer, _recorder, _webhook_manager, _ws_manager, _start_time

    _webhook_manager = WebhookManager()
    _ws_manager = WebSocketManager()

    def _on_state_change(old_state, new_state):
        _webhook_manager.check_and_fire(old_state, new_state)

    def _on_state_update(state):
        """Broadcast every state update to WebSocket clients."""
        if _ws_manager is not None and _ws_manager.connection_count > 0:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_ws_manager.broadcast(state))
            except RuntimeError:
                # No running event loop (e.g. called from a non-async thread).
                # Schedule on a new loop if needed -- but in practice the
                # FastAPI server loop will be running.
                pass

    if state_manager is not None:
        _state_manager = state_manager
        # Attach callbacks to pre-existing state manager
        _state_manager._on_state_change = _on_state_change
        _state_manager._on_state_update = _on_state_update
    else:
        _state_manager = StateManager(
            on_state_change=_on_state_change,
            on_state_update=_on_state_update,
        )

    _recorder = recorder
    _replayer = replayer

    if replayer is not None:
        # Replay mode: replayer already has a reference to the state_manager
        _reader = None
    else:
        _reader = BCIReader(
            state_manager=_state_manager, synthetic=synthetic, recorder=recorder,
            classifier=classifier,
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
        if _webhook_manager is not None:
            _webhook_manager.shutdown()

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

    @app.post("/webhooks")
    def register_webhook(body: WebhookRegistration) -> dict:
        """Register a new webhook for state-change notifications."""
        filters_dict = body.filters.model_dump(exclude_none=True) if body.filters else None
        wh_id = _webhook_manager.register(url=body.url, filters=filters_dict or None)
        return {"id": wh_id, "registered": True}

    @app.get("/webhooks")
    def list_webhooks() -> list[WebhookInfo]:
        """List all registered webhooks."""
        return [WebhookInfo(**wh) for wh in _webhook_manager.list()]

    @app.delete("/webhooks/{webhook_id}")
    def delete_webhook(webhook_id: str) -> dict:
        """Unregister a webhook by id."""
        removed = _webhook_manager.unregister(webhook_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Webhook not found")
        return {"deleted": True}

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        """Accept a WebSocket connection and stream BCI state updates."""
        await _ws_manager.connect(ws)
        try:
            # Send the current state immediately on connect
            current = _state_manager.get_state()
            if current is not None:
                await ws.send_json(current)

            # Keep the connection alive -- wait for client messages (pings)
            while True:
                # recv with a timeout so we can detect disconnects
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.debug("WebSocket connection error", exc_info=True)
        finally:
            _ws_manager.disconnect(ws)

    @app.get("/ws/info")
    def ws_info() -> dict:
        """Return WebSocket connection info for monitoring."""
        return {"connections": _ws_manager.connection_count}

    return app
