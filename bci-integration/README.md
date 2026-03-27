# BCI-to-OpenClaw Integration Prototype

Connects Galea BCI hardware (via BrainFlow SDK) to the OpenClaw AI agent platform for real-time brain-state-aware AI interactions.

## Architecture (v2)

A 3-component system that integrates with OpenClaw natively:

1. **Signal Processor + State Server** (Python) -- BrainFlow data acquisition, DSP (Welch's method band powers), heuristic brain state classification, FastAPI HTTP server
2. **OpenClaw BCI Plugin** (TypeScript) -- `before_prompt_build` hook that injects brain state into LLM context
3. **BCI SKILL.md** -- Teaches the agent how to interpret and adapt to brain states

See [docs/architecture.md](docs/architecture.md) for full details.

## Schemas

| Schema | Purpose |
|--------|---------|
| `bci_stream.schema.json` | Raw BCI packets from BrainFlow |
| `processed_features.schema.json` | Extracted frequency-domain features |
| `bci_state.schema.json` | LLM-readable brain state (injected via plugin) |
| `state_server_api.schema.json` | HTTP API contract (GET /state, GET /health) |

## Quick Start

```bash
# 1. Start the signal processor (synthetic mode, no hardware needed)
cd signal-processor
pip install -e .
python -m src --synthetic

# 2. Install the OpenClaw plugin
cd openclaw-plugin
npm install
# Add to OpenClaw's plugin config

# 3. Copy skill/SKILL.md to your OpenClaw skills directory
```

## Tech Stack

- Python: BrainFlow, FastAPI, NumPy, SciPy, Pydantic
- TypeScript: OpenClaw Plugin SDK
- Transport: HTTP/JSON (localhost)
