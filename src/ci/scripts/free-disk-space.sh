#!/bin/bash
set -euo pipefail

script_dir=$(dirname "$0")

if [[ "${RUNNER_OS:-}" == "Windows" ]]; then
    pwsh $script_dir/free-disk-space-windows.ps1
else
    $script_dir/free-disk-space-linux.sh
fi
