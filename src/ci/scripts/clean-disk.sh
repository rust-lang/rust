#!/bin/bash
# This script deletes some of the Azure-provided artifacts. We don't use these,
# and disk space is at a premium on our builders.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

# All the Linux builds happen inside Docker.
if isLinux; then
    # 6.7GB
    sudo rm -rf /opt/ghc
    # 16GB
    sudo rm -rf /usr/share/dotnet
fi
