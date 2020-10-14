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

MINGW_ARCHIVE_32="i686-6.3.0-release-posix-dwarf-rt_v5-rev2.7z"
MINGW_ARCHIVE_64="x86_64-6.3.0-release-posix-seh-rt_v5-rev2.7z"

if isWindows; then
    case "${CI_JOB_NAME}" in
        *i686*)
            bits=32
            arch=i686
            mingw_archive="${MINGW_ARCHIVE_32}"
            ;;
        *x86_64*)
            bits=64
            arch=x86_64
            mingw_archive="${MINGW_ARCHIVE_64}"
            ;;
        *aarch64*)
            # aarch64 is a cross-compiled target. Use the x86_64
            # mingw, since that's the host architecture.
            bits=64
            arch=x86_64
            mingw_archive="${MINGW_ARCHIVE_64}"
            ;;
        *)
            echo "src/ci/scripts/install-mingw.sh can't detect the builder's architecture"
            echo "please tweak it to recognize the builder named '${CI_JOB_NAME}'"
            exit 1
            ;;
    esac

    if [[ "${CUSTOM_MINGW-0}" -ne 1 ]]; then
        pacman -S --noconfirm --needed mingw-w64-$arch-toolchain mingw-w64-$arch-cmake \
            mingw-w64-$arch-gcc \
            mingw-w64-$arch-python # the python package is actually for python3
        ciCommandAddPath "$(ciCheckoutPath)/msys2/mingw${bits}/bin"
    else
        mingw_dir="mingw${bits}"

        curl -o mingw.7z "${MIRRORS_BASE}/${mingw_archive}"
        7z x -y mingw.7z > /dev/null
        curl -o "${mingw_dir}/bin/gdborig.exe" "${MIRRORS_BASE}/2017-04-20-${bits}bit-gdborig.exe"
        ciCommandAddPath "$(pwd)/${mingw_dir}/bin"
    fi
fi
