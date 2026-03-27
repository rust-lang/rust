# BCI-to-OpenClaw Integration Prototype

Connects Galea BCI hardware (via BrainFlow SDK) to the OpenClaw model integration layer for real-time brain-state classification.

## Architecture

A 5-component pipeline: BCI Reader -> Feature Extractor -> OpenClaw Gateway -> Model Backend, with a Monitor UI.

See [docs/architecture.md](docs/architecture.md) for full details.

## Schemas

All inter-component communication uses JSON validated against schemas in `schemas/`:

| Schema | Purpose |
|--------|---------|
| `bci_stream.schema.json` | Raw BCI packets from Galea/BrainFlow |
| `processed_features.schema.json` | Extracted frequency-domain features |
| `model_input.schema.json` | Request to OpenClaw model layer |
| `model_output.schema.json` | Response with classified brain state |

## Task Breakdown

See [docs/task-breakdown.md](docs/task-breakdown.md) for implementation plan (9 tasks, ~23h).

## Tech Stack

Python, BrainFlow, ZeroMQ, Flask, scikit-learn, NumPy/SciPy, rich, jsonschema
