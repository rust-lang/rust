#!/bin/bash
set -ex

if command -v mdbook >/dev/null 2>&1; then
    cargo install mdbook --vers "0.0.28"
fi

if command -v ghp-import >/dev/null 2>&1; then
    pip install ghp-import
fi
