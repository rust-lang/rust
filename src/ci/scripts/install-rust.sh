#!/bin/bash

# The Arm64 Windows Runner does not have Rust already installed
# https://github.com/actions/partner-runner-images/issues/77

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if [[ "${CI_JOB_NAME}" = *aarch64* ]] && isWindows; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y -q --default-host aarch64-pc-windows-msvc
    ciCommandAddPath "${USERPROFILE}/.cargo/bin"
fi
