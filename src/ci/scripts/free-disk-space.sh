#!/bin/bash
set -euo pipefail

script_dir=$(dirname "$0")

if [[ "${RUNNER_OS:-}" == "Windows" ]]; then
    python3 "$script_dir/free-disk-space-windows-start.py"
else
    $script_dir/free-disk-space-linux.sh
fi
