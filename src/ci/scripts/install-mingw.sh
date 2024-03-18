#!/bin/bash
# If we need to download a custom MinGW, do so here and set the path
# appropriately.
#
# Otherwise install MinGW through `pacman`

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

MINGW_ARCHIVE_32="i686-w64-mingw32.tar.zst"
MINGW_ARCHIVE_64="x86_64-w64-mingw32.tar.zst"

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

    if [[ "${CUSTOM_MINGW:-0}" == 0 ]]; then
        pacboy -S --noconfirm toolchain:p
        # According to the comment in the Windows part of install-clang.sh, in the future we might
        # want to do this instead:
        # pacboy -S --noconfirm clang:p ...
    else
        url="https://github.com/mati865/mingw-build/releases/download/v0.0.11"
        curl -L -o mingw.tar.zst "${url}/${mingw_archive}"
        tar -xf mingw.tar.zst
        ciCommandAddPath "$(pwd)/${arch}-w64-mingw32/bin"
        ciCommandAddPath "$(pwd)/${arch}-w64-mingw32/${arch}-w64-mingw32/bin"
    fi
fi
