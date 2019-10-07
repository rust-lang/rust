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
    # FIXME(#65767): workaround msys bug, step 1
    arch=i686
    if [ "$MSYS_BITS" = "64" ]; then
      arch=x86_64
    fi
    curl -O "${MIRRORS_BASE}/msys2-repo/mingw/$arch/mingw-w64-$arch-ca-certificates-20180409-1-any.pkg.tar.xz"

    choco install msys2 --params="/InstallDir:${SYSTEM_WORKFOLDER}/msys2 /NoPath" -y --no-progress
    mkdir -p "${SYSTEM_WORKFOLDER}/msys2/home/${USERNAME}"

    ciCommandAddPath "${SYSTEM_WORKFOLDER}/msys2/usr/bin"
    export PATH="${SYSTEM_WORKFOLDER}/msys2/usr/bin"

    pacman -S --noconfirm --needed base-devel ca-certificates make diffutils tar

    # FIXME(#65767): workaround msys bug, step 2
    pacman -U --noconfirm --noprogressbar mingw-w64-$arch-ca-certificates-20180409-1-any.pkg.tar.xz
    rm mingw-w64-$arch-ca-certificates-20180409-1-any.pkg.tar.xz

    # Make sure we use the native python interpreter instead of some msys equivalent
    # one way or another. The msys interpreters seem to have weird path conversions
    # baked in which break LLVM's build system one way or another, so let's use the
    # native version which keeps everything as native as possible.
    cp C:/Python27amd64/python.exe C:/Python27amd64/python2.7.exe
    ciCommandAddPath "C:\\Python27amd64"
fi
