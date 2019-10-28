#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    pacman -S --noconfirm --needed base-devel ca-certificates make diffutils tar

    # FIXME(#65767): workaround msys bug, step 2
    arch=i686
    if [ "$MSYS_BITS" = "64" ]; then
      arch=x86_64
    fi
    pacman -U --noconfirm --noprogressbar mingw-w64-$arch-ca-certificates-20180409-1-any.pkg.tar.xz
    rm mingw-w64-$arch-ca-certificates-20180409-1-any.pkg.tar.xz

    # Make sure we use the native python interpreter instead of some msys equivalent
    # one way or another. The msys interpreters seem to have weird path conversions
    # baked in which break LLVM's build system one way or another, so let's use the
    # native version which keeps everything as native as possible.
    cp C:/Python27amd64/python.exe C:/Python27amd64/python2.7.exe
    ciCommandAddPath "C:\\Python27amd64"
fi
