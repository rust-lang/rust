#!/bin/bash
# For mingw builds use a vendored mingw.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

MINGW_ARCHIVE_32="i686-14.1.0-release-posix-dwarf-msvcrt-rt_v12-rev0.7z"
MINGW_ARCHIVE_64="x86_64-14.1.0-release-posix-seh-msvcrt-rt_v12-rev0.7z"

if isWindows && isKnownToBeMingwBuild; then
    case "${CI_JOB_NAME}" in
        *i686*)
            bits=32
            mingw_archive="${MINGW_ARCHIVE_32}"
            ;;
        *x86_64*)
            bits=64
            mingw_archive="${MINGW_ARCHIVE_64}"
            ;;
        *aarch64*)
            # aarch64 is a cross-compiled target. Use the x86_64
            # mingw, since that's the host architecture.
            bits=64
            mingw_archive="${MINGW_ARCHIVE_64}"
            ;;
        *)
            echo "src/ci/scripts/install-mingw.sh can't detect the builder's architecture"
            echo "please tweak it to recognize the builder named '${CI_JOB_NAME}'"
            exit 1
            ;;
    esac

    mingw_dir="mingw${bits}"

    curl -o mingw.7z "${MIRRORS_BASE}/${mingw_archive}"
    7z x -y mingw.7z > /dev/null
    ciCommandAddPath "$(pwd)/${mingw_dir}/bin"
fi
