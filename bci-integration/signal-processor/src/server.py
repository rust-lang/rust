"""FastAPI HTTP server for BCI state.

Endpoints:
    GET /state  - Current brain state (state_server_api state_response)
    GET /health - Server health (state_server_api health_response)
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from . import config
from .brainflow_reader import BCIReader
from .state_manager import StateManager

logger = logging.getLogger(__name__)

# Module-level state shared between lifespan and endpoints
state_manager = StateManager()
reader: BCIReader | None = None
_start_time: float = 0.0
_synthetic: bool = True


def configure(synthetic: bool = True) -> None:
    """Configure the server before startup. Must be called before app lifespan."""
    global _synthetic
    _synthetic = synthetic


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage BrainFlow reader lifecycle."""
    global reader, _start_time
    _start_time = time.time()

    reader = BCIReader(state_manager=state_manager, synthetic=_synthetic)
    try:
        reader.start()
        logger.info("BCI reader started during server startup")
    except Exception:
        logger.exception("Failed to start BCI reader")
        reader = None

    yield

    if reader is not None:
        reader.stop()
        logger.info("BCI reader stopped during server shutdown")


app = FastAPI(title="BCI State Server", version="0.1.0", lifespan=lifespan)


@app.get("/state")
def get_state() -> dict[str, Any]:
    """Return current BCI brain state.

    Conforms to state_server_api.schema.json#/definitions/state_response.
    """
    now_ms = int(time.time() * 1000)

    bci_state = state_manager.get_state()
    if bci_state is None:
        return {
            "available": False,
            "timestamp_unix_ms": now_ms,
            "error": "No BCI data received yet",
        }

    staleness = bci_state.get("staleness_ms", 0)

    # If data is too stale, mark as degraded
    if staleness > config.STALE_TIMEOUT_MS:
        bci_state["signal_quality"] = 0.0
        bci_state["state"]["primary"] = "unknown"
        bci_state["state"]["confidence"] = 0.0
        bci_state["natural_language_summary"] = (
            f"User brain state: UNKNOWN (BCI data is stale, last update {staleness}ms ago)"
        )

    return {
        "available": True,
        "timestamp_unix_ms": now_ms,
        "bci_state": bci_state,
    }


@app.get("/health")
def get_health() -> dict[str, Any]:
    """Return server health status.

    Conforms to state_server_api.schema.json#/definitions/health_response.
    """
    uptime = time.time() - _start_time
    device_connected = reader is not None and reader.is_running
    last_update = state_manager.last_update_ms
    session_id = reader.session_id if reader else None

    # Determine status
    if not state_manager.has_data:
        status = "no_signal"
    else:
        bci_state = state_manager.get_state()
        staleness = bci_state.get("staleness_ms", 0) if bci_state else 0
        sig_quality = bci_state.get("signal_quality", 0) if bci_state else 0

        if staleness > config.DEGRADED_STALE_MS or sig_quality < config.DEGRADED_QUALITY_THRESHOLD:
            status = "degraded"
        else:
            status = "ok"

    return {
        "status": status,
        "uptime_seconds": round(uptime, 2),
        "device_connected": device_connected,
        "session_id": session_id,
        "last_update_unix_ms": last_update,
    }
