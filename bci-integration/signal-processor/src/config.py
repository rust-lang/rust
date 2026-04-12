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

# Pause Detection
CLENCH_THRESHOLD_FACTOR: float = float(os.environ.get("BCI_CLENCH_THRESHOLD_FACTOR", "3.0"))
CLENCH_WINDOW_S: float = float(os.environ.get("BCI_CLENCH_WINDOW_S", "2.0"))
CLENCH_MIN_DURATION_MS: int = int(os.environ.get("BCI_CLENCH_MIN_DURATION_MS", "100"))
DROWSINESS_WINDOW_SIZE: int = int(os.environ.get("BCI_DROWSINESS_WINDOW_SIZE", "20"))
DROWSINESS_ATTENTION_THRESHOLD: float = float(os.environ.get("BCI_DROWSINESS_ATTENTION_THRESHOLD", "0.25"))
HEADSET_REMOVED_WINDOW_SIZE: int = int(os.environ.get("BCI_HEADSET_REMOVED_WINDOW_SIZE", "8"))
HEADSET_REMOVED_QUALITY_THRESHOLD: float = float(os.environ.get("BCI_HEADSET_REMOVED_QUALITY_THRESHOLD", "0.05"))

# ML Classifier
MODEL_PATH: str | None = os.environ.get("BCI_MODEL_PATH", None)

# Deep Learning Classifier (EEGNet)
DEEP_MODEL_PATH: str | None = os.environ.get("BCI_DEEP_MODEL_PATH", None)

# Webhooks
WEBHOOK_COOLDOWN_MS: int = int(os.environ.get("BCI_WEBHOOK_COOLDOWN_MS", "1000"))
WEBHOOK_TIMEOUT_S: int = int(os.environ.get("BCI_WEBHOOK_TIMEOUT_S", "5"))

# WebSocket
WS_HEARTBEAT_INTERVAL_S: int = int(os.environ.get("BCI_WS_HEARTBEAT_INTERVAL_S", "30"))
