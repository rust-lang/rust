"""WebSocket connection manager for broadcasting BCI state updates.

Maintains a set of active WebSocket connections and provides efficient
broadcast (serialize once, send to all).
"""

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages active WebSocket connections and broadcasts state updates."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await ws.accept()
        self._connections.add(ws)
        logger.info(
            "WebSocket client connected (total: %d)", len(self._connections)
        )

    def disconnect(self, ws: WebSocket) -> None:
        """Remove a WebSocket connection from the active set."""
        self._connections.discard(ws)
        logger.info(
            "WebSocket client disconnected (total: %d)", len(self._connections)
        )

    async def broadcast(self, bci_state: dict[str, Any]) -> None:
        """Send a JSON-encoded state to all connected clients.

        Serializes once and sends the raw text to every connection.
        Disconnected clients are removed automatically.
        """
        if not self._connections:
            return

        payload = json.dumps(bci_state)
        dead: list[WebSocket] = []

        async def _send(ws: WebSocket) -> None:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        tasks = [asyncio.create_task(_send(ws)) for ws in self._connections]
        await asyncio.gather(*tasks)

        for ws in dead:
            self._connections.discard(ws)
            logger.info(
                "Removed dead WebSocket client (total: %d)",
                len(self._connections),
            )

    @property
    def connection_count(self) -> int:
        """Return the number of active WebSocket connections."""
        return len(self._connections)
