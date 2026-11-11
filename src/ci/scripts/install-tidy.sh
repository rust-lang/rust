#!/bin/bash
# This script downloads and installs the tidy binary from Homebrew.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

# Only the macOS arm64/aarch64 GitHub Actions runner needs to have tidy
# installed; other platforms have it preinstalled.

if isMacOS; then
    platform=$(uname -m)
    case $platform in
        x86_64)
            ;;
        arm64)
            brew install tidy-html5
            ;;
        *)
            echo "unsupported architecture: ${platform}"
            exit 1
    esac
fi
