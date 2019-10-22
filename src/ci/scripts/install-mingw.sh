#!/bin/bash
# If we need to download a custom MinGW, do so here and set the path
# appropriately.
#
# Here we also do a pretty heinous thing which is to mangle the MinGW
# installation we just downloaded. Currently, as of this writing, we're using
# MinGW-w64 builds of gcc, and that's currently at 6.3.0. We use 6.3.0 as it
# appears to be the first version which contains a fix for #40546, builds
# randomly failing during LLVM due to ar.exe/ranlib.exe failures.
#
# Unfortunately, though, 6.3.0 *also* is the first version of MinGW-w64 builds
# to contain a regression in gdb (#40184). As a result if we were to use the
# gdb provided (7.11.1) then we would fail all debuginfo tests.
#
# In order to fix spurious failures (pretty high priority) we use 6.3.0. To
# avoid disabling gdb tests we download an *old* version of gdb, specifically
# that found inside the 6.2.0 distribution. We then overwrite the 6.3.0 gdb
# with the 6.2.0 gdb to get tests passing.
#
# Note that we don't literally overwrite the gdb.exe binary because it appears
# to just use gdborig.exe, so that's the binary we deal with instead.
#
# Otherwise install MinGW through `pacman`

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    if [[ -z "${MINGW_URL+x}" ]]; then
        arch=i686
        if [ "$MSYS_BITS" = "64" ]; then
          arch=x86_64
        fi
        pacman -S --noconfirm --needed mingw-w64-$arch-toolchain mingw-w64-$arch-cmake \
            mingw-w64-$arch-gcc mingw-w64-$arch-python2
        ciCommandAddPath "${SYSTEM_WORKFOLDER}/msys2/mingw${MSYS_BITS}/bin"
    else
        curl -o mingw.7z "${MINGW_URL}/${MINGW_ARCHIVE}"
        7z x -y mingw.7z > /dev/null
        curl -o "${MINGW_DIR}/bin/gdborig.exe" "${MINGW_URL}/2017-04-20-${MSYS_BITS}bit-gdborig.exe"
        ciCommandAddPath "$(pwd)/${MINGW_DIR}/bin"
    fi
fi
