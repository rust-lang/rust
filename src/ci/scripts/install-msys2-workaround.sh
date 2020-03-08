#!/bin/bash
# Downloads and "installs" msys2 and MinGW by downloading and extracting a
# pre-installed copy from our mirrors. This is a workaround for repo.msys2.org
# being offline, and it should be reverted as soon as it gets live again.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

WORKAROUND_TARBALL_DATE="2020-03-08"
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
        *)
            echo "src/ci/scripts/install-mingw.sh can't detect the builder's architecture"
            echo "please tweak it to recognize the builder named '${CI_JOB_NAME}'"
            exit 1
            ;;
    esac

    if [[ "${CUSTOM_MINGW-0}" -ne 1 ]]; then
        curl -o msys2.7z "${MIRRORS_BASE}/${WORKAROUND_TARBALL_DATE}-msys2-msvc-${arch}.7z"
        7z x -y msys2.7z > /dev/null

        ciCommandAddPath "$(ciCheckoutPath)/msys2/mingw${bits}/bin"
    else
        mingw_dir="mingw${bits}"

        curl -o mingw-base.7z "${MIRRORS_BASE}/${WORKAROUND_TARBALL_DATE}-msys2-mingw-base.7z"
        7z x -y mingw-base.7z > /dev/null
        curl -o mingw.7z "${MIRRORS_BASE}/${mingw_archive}"
        7z x -y mingw.7z > /dev/null
        curl -o "${mingw_dir}/bin/gdborig.exe" "${MIRRORS_BASE}/2017-04-20-${bits}bit-gdborig.exe"
        ciCommandAddPath "$(pwd)/${mingw_dir}/bin"
    fi

    mkdir -p "$(ciCheckoutPath)/msys2/home/${USERNAME}"
    ciCommandAddPath "$(ciCheckoutPath)/msys2/usr/bin"

    # Make sure we use the native python interpreter instead of some msys equivalent
    # one way or another. The msys interpreters seem to have weird path conversions
    # baked in which break LLVM's build system one way or another, so let's use the
    # native version which keeps everything as native as possible.
    cp C:/Python27amd64/python.exe C:/Python27amd64/python2.7.exe
    ciCommandAddPath "C:\\Python27amd64"
fi
