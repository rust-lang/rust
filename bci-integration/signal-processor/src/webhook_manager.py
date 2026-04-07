"""Webhook manager for notifying external services on brain state changes.

Stores registered webhooks in-memory and fires HTTP POSTs on state
transitions, with per-URL cooldown and filter matching.
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from . import config

logger = logging.getLogger(__name__)


class _Webhook:
    """Internal representation of a registered webhook."""

    __slots__ = ("id", "url", "filters", "created_at", "last_fired_at")

    def __init__(self, url: str, filters: dict[str, Any] | None = None) -> None:
        self.id: str = str(uuid.uuid4())
        self.url: str = url
        self.filters: dict[str, Any] | None = filters
        self.created_at: int = int(time.time() * 1000)
        self.last_fired_at: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "filters": self.filters,
            "created_at": self.created_at,
        }


class WebhookManager:
    """Thread-safe webhook registration and delivery."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._webhooks: list[_Webhook] = []
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._client = httpx.Client(timeout=config.WEBHOOK_TIMEOUT_S)

    def register(self, url: str, filters: dict[str, Any] | None = None) -> str:
        """Register a webhook and return its id."""
        wh = _Webhook(url=url, filters=filters)
        with self._lock:
            self._webhooks.append(wh)
        logger.info("Registered webhook %s -> %s", wh.id, url)
        return wh.id

    def unregister(self, webhook_id: str) -> bool:
        """Remove a webhook by id. Returns True if found and removed."""
        with self._lock:
            for i, wh in enumerate(self._webhooks):
                if wh.id == webhook_id:
                    self._webhooks.pop(i)
                    logger.info("Unregistered webhook %s", webhook_id)
                    return True
        return False

    def list(self) -> list[dict[str, Any]]:
        """Return list of registered webhooks as dicts."""
        with self._lock:
            return [wh.to_dict() for wh in self._webhooks]

    def check_and_fire(
        self, old_state: dict[str, Any], new_state: dict[str, Any]
    ) -> None:
        """Check filters and fire matching webhooks for a state transition.

        Called from the acquisition thread. Webhook delivery is offloaded
        to a thread pool so this method returns quickly.
        """
        old_primary = old_state.get("state", {}).get("primary", "unknown")
        new_primary = new_state.get("state", {}).get("primary", "unknown")

        if old_primary == new_primary:
            return

        now_ms = int(time.time() * 1000)
        payload = {
            "event": "state_change",
            "previous_state": old_primary,
            "new_state": new_primary,
            "bci_state": new_state,
            "timestamp_unix_ms": now_ms,
        }

        with self._lock:
            targets = []
            for wh in self._webhooks:
                if not self._matches_filters(wh, new_primary, new_state):
                    continue
                if (
                    wh.last_fired_at is not None
                    and (now_ms - wh.last_fired_at) < config.WEBHOOK_COOLDOWN_MS
                ):
                    logger.debug(
                        "Webhook %s cooldown active, skipping", wh.id
                    )
                    continue
                wh.last_fired_at = now_ms
                targets.append((wh.id, wh.url))

        for wh_id, url in targets:
            self._executor.submit(self._deliver, wh_id, url, payload)

    def shutdown(self) -> None:
        """Shut down the thread pool and HTTP client."""
        self._executor.shutdown(wait=False)
        self._client.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_filters(
        wh: _Webhook, new_primary: str, new_state: dict[str, Any]
    ) -> bool:
        """Return True if the webhook's filters match the transition."""
        filters = wh.filters
        if filters is None:
            return True  # no filters = match everything

        # state_is filter
        state_is = filters.get("state_is")
        if state_is is not None and new_primary != state_is:
            return False

        scores = new_state.get("scores", {})

        # score_below filter: {"score_name": threshold}
        score_below = filters.get("score_below")
        if score_below is not None:
            for score_name, threshold in score_below.items():
                value = scores.get(score_name)
                if value is None or value >= threshold:
                    return False

        # score_above filter: {"score_name": threshold}
        score_above = filters.get("score_above")
        if score_above is not None:
            for score_name, threshold in score_above.items():
                value = scores.get(score_name)
                if value is None or value <= threshold:
                    return False

        return True

    def _deliver(self, wh_id: str, url: str, payload: dict[str, Any]) -> None:
        """Send the webhook payload via HTTP POST. Never raises."""
        try:
            resp = self._client.post(url, json=payload)
            logger.info(
                "Webhook %s delivered to %s (status %d)",
                wh_id,
                url,
                resp.status_code,
            )
        except Exception:
            logger.exception("Webhook %s delivery to %s failed", wh_id, url)
