.PHONY: install test run-synthetic test-python test-plugin build-plugin clean train

# Python signal processor
install:
	cd signal-processor && pip install -e ".[dev]"

test-python:
	cd signal-processor && python -m pytest tests/ -v

run-synthetic:
	cd signal-processor && python -m src --synthetic

# OpenClaw plugin
build-plugin:
	cd openclaw-plugin && npm run build

test-plugin:
	cd openclaw-plugin && npm test

# All tests
test: test-python test-plugin

# Train ML classifier
train:
	cd signal-processor && python -m src.train_model --generate-synthetic --seconds 30 --output models/brain_state_rf.joblib

# Clean artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf signal-processor/dist signal-processor/build
	rm -rf openclaw-plugin/dist openclaw-plugin/node_modules
