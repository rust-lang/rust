#!/bin/bash
set -ex

if command -v mdbook >/dev/null 2>&1; then
    echo "installing mdbook"
    cargo install mdbook --vers "0.0.28"
else
    echo "mdbook already installed at $(which mdbook)"
fi

if command -v ghp-import >/dev/null 2>&1; then
    echo "installing ghp-import"
    pip install ghp-import
else 
    echo "ghp-import already installed at $(which ghp-import)"
fi
