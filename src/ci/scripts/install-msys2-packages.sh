#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    pacman -S --noconfirm --needed base-devel ca-certificates make diffutils tar \
        binutils

    # Make sure we use the native python interpreter instead of some msys equivalent
    # one way or another. The msys interpreters seem to have weird path conversions
    # baked in which break LLVM's build system one way or another, so let's use the
    # native version which keeps everything as native as possible.
    python_home="C:/hostedtoolcache/windows/Python/2.7.17/x64"
    cp "${python_home}/python.exe" "${python_home}/python2.7.exe"
    ciCommandAddPath "C:\\hostedtoolcache\\windows\\Python\\2.7.17\\x64"
fi
