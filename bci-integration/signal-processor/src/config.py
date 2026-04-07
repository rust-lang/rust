"""Configuration constants loaded from environment variables with defaults."""

import os

# Server
PORT: int = int(os.environ.get("BCI_PORT", "7680"))
HOST: str = os.environ.get("BCI_HOST", "127.0.0.1")

# DSP
SAMPLE_RATE: int = int(os.environ.get("BCI_SAMPLE_RATE", "250"))
WINDOW_SIZE_MS: int = int(os.environ.get("BCI_WINDOW_SIZE_MS", "1000"))
WINDOW_STEP_MS: int = int(os.environ.get("BCI_WINDOW_STEP_MS", "250"))

# Derived
WINDOW_SIZE_SAMPLES: int = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)

# Thresholds
ARTIFACT_AMPLITUDE_UV: float = float(os.environ.get("BCI_ARTIFACT_AMPLITUDE_UV", "100.0"))
STALE_THRESHOLD_MS: int = int(os.environ.get("BCI_STALE_THRESHOLD_MS", "2000"))
STALE_WARNING_MS: int = int(os.environ.get("BCI_STALE_WARNING_MS", "5000"))

# Webhooks
WEBHOOK_COOLDOWN_MS: int = int(os.environ.get("BCI_WEBHOOK_COOLDOWN_MS", "1000"))
WEBHOOK_TIMEOUT_S: int = int(os.environ.get("BCI_WEBHOOK_TIMEOUT_S", "5"))
