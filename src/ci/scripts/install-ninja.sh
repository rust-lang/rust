#!/bin/bash
# Note that this is originally from the github releases patch of Ninja

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

# Check and set default values for variables
: "${MIRRORS_BASE:?MIRRORS_BASE is not set}"
: "${RUST_CONFIGURE_ARGS:=}"

if isWindows; then
    mkdir -p ninja
    curl -fLo ninja.zip "${MIRRORS_BASE}/2024-03-28-v1.11.1-ninja-win.zip"
    7z x -oninja ninja.zip
    rm ninja.zip
    ciCommandSetEnv "RUST_CONFIGURE_ARGS" "${RUST_CONFIGURE_ARGS} --enable-ninja"
    ciCommandAddPath "$(cygpath -m "$(pwd)/ninja")"
elif isMacOS; then
    brew install ninja
fi
