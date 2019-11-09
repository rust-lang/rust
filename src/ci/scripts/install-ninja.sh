#!/bin/bash
# Note that this is originally from the github releases patch of Ninja

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    mkdir ninja
    curl -o ninja.zip "${MIRRORS_BASE}/2017-03-15-ninja-win.zip"
    7z x -oninja ninja.zip
    rm ninja.zip
    ciCommandSetEnv "RUST_CONFIGURE_ARGS" "${RUST_CONFIGURE_ARGS} --enable-ninja"
    ciCommandAddPath "$(pwd)/ninja"
fi
