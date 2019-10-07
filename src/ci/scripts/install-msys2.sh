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
    export PATH="${SYSTEM_WORKFOLDER}/msys2/usr/bin"

    pacman -S --noconfirm --needed base-devel ca-certificates make diffutils tar

    # Make sure we use the native python interpreter instead of some msys equivalent
    # one way or another. The msys interpreters seem to have weird path conversions
    # baked in which break LLVM's build system one way or another, so let's use the
    # native version which keeps everything as native as possible.
    cp C:/Python27amd64/python.exe C:/Python27amd64/python2.7.exe
    ciCommandAddPath "C:\\Python27amd64"
fi
