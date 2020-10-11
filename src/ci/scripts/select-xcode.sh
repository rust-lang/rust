#!/bin/bash
# This script selects the Xcode instance to use.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isMacOS; then
    if [[ -s "${SELECT_XCODE-}" ]]; then
        sudo xcode-select -s "${SELECT_XCODE}"
    fi
fi
