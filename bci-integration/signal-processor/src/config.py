"""Configuration constants loaded from environment variables with defaults."""

import os


HOST: str = os.environ.get("BCI_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("BCI_PORT", "7680"))
SAMPLE_RATE: int = int(os.environ.get("BCI_SAMPLE_RATE", "250"))
WINDOW_SIZE_MS: int = int(os.environ.get("BCI_WINDOW_SIZE_MS", "1000"))
WINDOW_STEP_MS: int = int(os.environ.get("BCI_WINDOW_STEP_MS", "250"))

# Derived
WINDOW_SIZE_SAMPLES: int = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)
WINDOW_STEP_SAMPLES: int = int(SAMPLE_RATE * WINDOW_STEP_MS / 1000)

# Thresholds
ARTIFACT_AMPLITUDE_UV: float = float(os.environ.get("BCI_ARTIFACT_AMPLITUDE_UV", "100.0"))
STALE_TIMEOUT_MS: int = int(os.environ.get("BCI_STALE_TIMEOUT_MS", "2000"))
DEGRADED_STALE_MS: int = int(os.environ.get("BCI_DEGRADED_STALE_MS", "5000"))
DEGRADED_QUALITY_THRESHOLD: float = 0.3
