#!/bin/bash
# This script installs sccache on the local machine. Note that we don't install
# sccache on Linux since it's installed elsewhere through all the containers.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isMacOS; then
    curl -fo /usr/local/bin/sccache \
      "${MIRRORS_BASE}/2025-02-24-sccache-v0.10.0-x86_64-apple-darwin"
    chmod +x /usr/local/bin/sccache
elif isWindows; then
    mkdir -p sccache
    curl -fo sccache/sccache.exe \
      "${MIRRORS_BASE}/2025-02-24-sccache-v0.10.0-x86_64-pc-windows-msvc.exe"
    ciCommandAddPath "$(cygpath -m "$(pwd)/sccache")"
fi

# FIXME: we should probably install sccache outside the containers and then
# mount it inside the containers so we can centralize all installation here.
