#!/bin/bash
# This script sets some environment variables for our custom v810 build

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

ciCommandSetEnv "CI_JOB_NAME" "build"
ciCommandSetEnv "RUST_CONFIGURE_ARGS" ""

if isMacOS; then
    ciCommandSetEnv "SELECT_XCODE" "/Applications/Xcode_15.4.app"
    ciCommandSetEnv "USE_XCODE_CLANG" "1"
    # Aarch64 tooling only needs to support macOS 11.0 and up as nothing else
    # supports the hardware, so only need to test it there.
    ciCommandSetEnv "MACOSX_DEPLOYMENT_TARGET" "11.0"
    ciCommandSetEnv "MACOSX_STD_DEPLOYMENT_TARGET" "11.0"
fi
