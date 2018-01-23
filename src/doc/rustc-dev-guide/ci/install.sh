#!/bin/bash
set -ex

if command -v mdbook >/dev/null 2>&1; then
    echo "mdbook already installed at $(command -v mdbook)"
else
    echo "installing mdbook"
    cargo install mdbook --vers "0.0.28"
fi

if command -v ghp-import >/dev/null 2>&1; then
    echo "ghp-import already installed at $(which ghp-import)"
else
    echo "installing ghp-import"
    pip install --user ghp-import
fi
