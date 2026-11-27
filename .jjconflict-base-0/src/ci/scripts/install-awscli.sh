#!/bin/bash
# This script downloads and installs the awscli binaries directly from
# Amazon.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

AWS_VERSION="2.13.25"

# Only the macOS arm64/aarch64 GitHub Actions runner needs to have AWS
# installed; other platforms have it preinstalled.

if isMacOS; then
    platform=$(uname -m)
    case $platform in
        x86_64)
            ;;
        arm64)
            file="https://awscli.amazonaws.com/AWSCLIV2-${AWS_VERSION}.pkg"
            retry curl -f "${file}" -o "AWSCLIV2.pkg"
            sudo installer -pkg "AWSCLIV2.pkg" -target /
            ;;
        *)
            echo "unsupported architecture: ${platform}"
            exit 1
    esac
fi
