#!/bin/bash
# Download and install MSYS2, needed primarily for the test suite (run-make) but
# also used by the MinGW toolchain for assembling things.
#
# FIXME: we should probe the default azure image and see if we can use the MSYS2
# toolchain there. (if there's even one there). For now though this gets the job
# done.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    choco install msys2 --params="/InstallDir:${SYSTEM_WORKFOLDER}/msys2 /NoPath" -y --no-progress
    mkdir -p "${SYSTEM_WORKFOLDER}/msys2/home/${USERNAME}"

    ciCommandAddPath "${SYSTEM_WORKFOLDER}/msys2/usr/bin"
fi
