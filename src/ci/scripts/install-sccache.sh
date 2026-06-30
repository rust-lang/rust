#!/bin/bash
# This script installs sccache on the local machine. Note that we don't install
# sccache on Linux since it's installed elsewhere through all the containers.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isMacOS; then
    curl -fo sccache.tar.gz \
      "${MIRRORS_BASE}/2026-06-19-sccache-v0.16.0-x86_64-apple-darwin.tar.gz"
    tar -xvf sccache.tar.gz --strip-components 1
    mv sccache /usr/local/bin/sccache
    chmod +x /usr/local/bin/sccache
elif isWindows; then
    mkdir -p sccache
    curl -fo sccache/sccache.zip \
      "${MIRRORS_BASE}/2026-06-19-sccache-v0.16.0-x86_64-pc-windows-msvc.zip"
    unzip -j sccache/sccache.zip sccache-v0.16.0-x86_64-pc-windows-msvc/sccache.exe -d sccache
    ls sccache
    ciCommandAddPath "$(cygpath -m "$(pwd)/sccache")"
fi

# FIXME: we should probably install sccache outside the containers and then
# mount it inside the containers so we can centralize all installation here.
