#!/bin/bash
# If we need to download a custom MinGW, do so here and set the path
# appropriately.
#
# Otherwise install MinGW through `pacman`

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

MINGW_ARCHIVE_32="i686-12.2.0-release-posix-dwarf-rt_v10-rev0.7z"
MINGW_ARCHIVE_64="x86_64-12.2.0-release-posix-seh-rt_v10-rev0.7z"

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
        ciCommandAddPath "$(pwd)/${mingw_dir}/bin"
    fi
fi
