#!/bin/bash
set -ex

cargo install cargo install mdbook --vers "0.0.28"

if command -v ghp-import >/dev/null 2>&1; then
    pip install ghp-import
fi
