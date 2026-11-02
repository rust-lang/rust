#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)

# dispatch according to the Github Runner
case "${RUNNER_OS:-}" in
    Linux)
        exec "${script_dir}/free-disk-space-linux.sh"
        ;;
    macOS)
        exec "${script_dir}/free-disk-space-mac.sh"
        ;;
    *)
        echo "::error:: freeing disk space not supported for: '${RUNNER_OS:-unset}'"
        exit 1
        ;;
esac
