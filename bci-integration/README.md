# BCI-to-OpenClaw Integration

A system that connects an EEG headset to the OpenClaw AI agent platform, enabling the LLM to be aware of your real-time brain state. A Python server reads brainwaves at 250Hz, classifies your cognitive state (focused, relaxed, stressed, drowsy), and injects a one-line summary into every AI response. The AI subtly adapts — shorter answers when you're tired, more detail when you're focused, gentler tone when you're stressed. Includes jaw-clench pause detection, session recording/replay, webhooks for proactive alerts, WebSocket streaming, and optional deep learning classification via EEGNet.

## Architecture

```
BrainFlow (250Hz EEG) --> DSP (Welch band powers) --> Classifier --> StateManager
   |                                                    |                |
   |  (or SessionReplayer)                    PauseDetector        FastAPI :7680
   |                                          (clench/drowsy)     /state /health
   |                                                              /ws /webhooks
   |                                                                   |
   +-- SessionRecorder (JSONL)                              +---------+---------+
                                                            |         |         |
                                                         Plugin    WS Client  Webhooks
                                                     (before_prompt   (any)   (Slack...)
                                                       _build)
                                                            |
                                                    OpenClaw Gateway
                                                            |
                                                      LLM (Claude, etc.)
```

Three custom components; everything else is handled by OpenClaw natively. See [docs/architecture.md](docs/architecture.md) for full details.

## Prerequisites

- Python >= 3.10
- Node.js >= 20 (for plugin)
- pip, npm

## Quick Start

```bash
cd bci-integration

# Install Python dependencies
make install

# Start the signal processor (synthetic EEG, no hardware needed)
make run-synthetic

# In another terminal, verify it works
curl http://127.0.0.1:7680/health
curl http://127.0.0.1:7680/state

# Run the interactive demo
./scripts/demo.sh
```

## Running Tests

```bash
make test          # Python + TypeScript tests
make test-python   # Python only (133 tests)
make test-plugin   # TypeScript only
```

## CLI Reference

```
python -m src [OPTIONS]

--port PORT           Server port (default: 7680)
--host HOST           Server host (default: 127.0.0.1)
--synthetic           Use fake EEG data (default: true)
--no-synthetic        Use real BCI hardware
--record FILE         Write brain states to JSONL while running
--replay FILE         Replay a recorded session instead of BrainFlow
--model PATH          Load trained Random Forest classifier (.joblib)
--deep-model PATH     Load trained EEGNet classifier (.pt)
--log-level LEVEL     Logging level (default: INFO)
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/state` | GET | Current brain state (primary state, scores, band powers, NL summary) |
| `/health` | GET | Server health (`ok`, `degraded`, `no_signal`) |
| `/pause` | GET | Pause detection state (`paused`, `reason`, `since_ms`) |
| `/ws` | WebSocket | Real-time state streaming; sends current state on connect |
| `/ws/info` | GET | WebSocket connection count |
| `/webhooks` | POST | Register a webhook (`{"url": "...", "filters": {...}}`) |
| `/webhooks` | GET | List all registered webhooks |
| `/webhooks/{id}` | DELETE | Remove a webhook |

### GET /state response

```json
{
  "available": true,
  "timestamp_unix_ms": 1712505600000,
  "bci_state": {
    "state": { "primary": "focused", "confidence": 0.72 },
    "scores": { "attention": 0.68, "relaxation": 0.31, "cognitive_load": 0.42 },
    "band_powers": { "delta": 12.4, "theta": 5.1, "alpha": 7.8, "beta": 15.2, "gamma": 2.1 },
    "signal_quality": 0.85,
    "natural_language_summary": "User brain state: FOCUSED (confidence: 0.72, ...)",
    "classification_source": "heuristic",
    "pause_event": null
  }
}
```

## Configuration

All environment variables have sensible defaults. See [.env.example](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `BCI_PORT` | `7680` | HTTP/WS server port |
| `BCI_HOST` | `127.0.0.1` | Server bind address |
| `BCI_SAMPLE_RATE` | `250` | EEG sampling rate (Hz) |
| `BCI_STALE_THRESHOLD_MS` | `2000` | State overridden to "unknown" after this |
| `BCI_MODEL_PATH` | (none) | Path to Random Forest .joblib model |
| `BCI_DEEP_MODEL_PATH` | (none) | Path to EEGNet .pt model |
| `BCI_CLENCH_THRESHOLD_FACTOR` | `3.0` | Gamma power must be Nx baseline for clench |
| `BCI_WEBHOOK_COOLDOWN_MS` | `1000` | Minimum ms between webhook fires per URL |
| `BCI_CORS_ORIGINS` | `*` | Comma-separated CORS allowed origins |
| `BCI_STATE_SERVER_URL` | `http://127.0.0.1:7680` | Plugin: state server URL |
| `BCI_USE_WEBSOCKET` | `false` | Plugin: use persistent WS instead of HTTP |

## OpenClaw Plugin Setup

```bash
# Build the plugin
cd openclaw-plugin && npm install && npm run build

# Register with OpenClaw (one of these):
openclaw plugins install -l /path/to/bci-integration/openclaw-plugin
# OR add to openclaw.json: { "plugins": [{ "path": "..." }] }

# Copy the skill so the agent knows how to interpret brain states
cp skill/SKILL.md ~/.openclaw/skills/bci-brain-state/SKILL.md
```

The plugin injects `natural_language_summary` into every AI turn via `before_prompt_build`. It also registers a `bci.status` tool for detailed JSON inspection.

## Recording and Replay

```bash
# Record a session while running
python -m src --synthetic --record session.jsonl

# Replay it later (deterministic, same states every time)
python -m src --replay session.jsonl
```

Sessions are JSONL files (one BCIState JSON object per line, 4 per second). Useful for demos, testing, and ML training data.

## ML Classifier

### Random Forest (from session recordings)

```bash
make train                    # Generate synthetic data + train
python -m src --model models/brain_state_rf.joblib
```

### EEGNet Deep Learning (from DREAMER dataset or synthetic)

```bash
make install-deep             # Install torch + braindecode
make train-deep               # Train on synthetic data
python -m src --deep-model models/eegnet_synthetic.pt

# Or train on DREAMER dataset (download from https://zenodo.org/records/546113)
python -m src.train_eegnet --input DREAMER.mat --output models/eegnet_dreamer.pt
```

The deep classifier operates on raw EEG (not pre-computed features). It overrides state classification while the heuristic still provides attention/relaxation/cognitive_load scores. `classification_source` in the response shows which classifier produced the state (`"heuristic"` or `"deep"`).

## Pause Detection

The system detects three pause triggers:

| Trigger | Type | Detection | Latency |
|---------|------|-----------|---------|
| **Jaw clench** (2x in 2s) | Deliberate | High-gamma (>30Hz) spike in temporal channels | <500ms |
| **Drowsiness** | Automatic | Sustained low attention (<0.25 for 5s) | ~5s |
| **Headset removed** | Automatic | Signal quality drops to ~0 for 2s | ~2s |

When triggered, `pause_event` appears in the BCIState and the plugin prepends `[PAUSE: ...]` or `[NOTICE: ...]` to the LLM context. Check `GET /pause` for current pause state.

## WebSocket Streaming

Connect to `ws://127.0.0.1:7680/ws` to receive real-time BCIState JSON pushes (~4/second). Current state is sent immediately on connect. Monitor connections via `GET /ws/info`.

For the OpenClaw plugin, set `BCI_USE_WEBSOCKET=true` to use a persistent WebSocket connection with auto-reconnect instead of HTTP polling.

## Webhook Integration

Register webhooks to get notified on brain state changes:

```bash
# Notify when user becomes drowsy
curl -X POST http://127.0.0.1:7680/webhooks \
  -H "Content-Type: application/json" \
  -d '{"url": "https://hooks.slack.com/...", "filters": {"state_is": "drowsy"}}'

# Notify when attention drops below 0.3
curl -X POST http://127.0.0.1:7680/webhooks \
  -d '{"url": "https://...", "filters": {"score_below": {"attention": 0.3}}}'

# List / delete
curl http://127.0.0.1:7680/webhooks
curl -X DELETE http://127.0.0.1:7680/webhooks/{id}
```

Webhook payload: `{"event": "state_change", "previous_state": "focused", "new_state": "drowsy", "bci_state": {...}}`.

## Schemas

| Schema | Purpose |
|--------|---------|
| `bci_state.schema.json` | Brain state object (what the LLM sees) |
| `state_server_api.schema.json` | HTTP API contract (GET /state, GET /health) |
| `bci_stream.schema.json` | Raw BCI packet format (design reference) |
| `processed_features.schema.json` | DSP feature format (design reference) |

## Limitations

- **No authentication** -- localhost only, no TLS. Do not expose to the internet.
- **Heuristic classifier** -- Threshold-based rules, not a trained model (unless you train one).
- **Synthetic data** -- Prototype tested with BrainFlow synthetic board, not real hardware.
- **Single user / single device** -- No multi-user or multi-device support.
- **No persistent storage** -- State is in-memory only. Webhooks are ephemeral.
