#!/bin/bash
# For mingw builds use a vendored mingw.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

MINGW_ARCHIVE_32="i686-14.1.0-release-posix-dwarf-msvcrt-rt_v12-rev0.7z"
MINGW_ARCHIVE_64="x86_64-14.1.0-release-posix-seh-msvcrt-rt_v12-rev0.7z"
LLVM_MINGW_ARCHIVE_AARCH64="llvm-mingw-20251104-ucrt-aarch64.zip"
LLVM_MINGW_ARCHIVE_X86_64="llvm-mingw-20251104-ucrt-x86_64.zip"

if isWindows && isKnownToBeMingwBuild; then
    case "${CI_JOB_NAME}" in
        *aarch64-llvm*)
            mingw_dir="clangarm64"
            mingw_archive="${LLVM_MINGW_ARCHIVE_AARCH64}"
            arch="aarch64"
            # Rustup defaults to AArch64 MSVC which has a hard time building Ring crate
            # for citool. MSVC jobs install special Clang build to solve that, but here
            # it would be an overkill. So we just use toolchain that doesn't have this
            # issue.
            rustup default stable-aarch64-pc-windows-gnullvm
            ;;
        *x86_64-llvm*)
            mingw_dir="clang64"
            mingw_archive="${LLVM_MINGW_ARCHIVE_X86_64}"
            arch="x86_64"
            ;;
        *i686*)
            mingw_dir="mingw32"
            mingw_archive="${MINGW_ARCHIVE_32}"
            ;;
        *x86_64*)
            mingw_dir="mingw64"
            mingw_archive="${MINGW_ARCHIVE_64}"
            ;;
        *aarch64*)
            echo "AArch64 Windows is not supported by GNU tools"
            exit 1
            ;;
        *)
            echo "src/ci/scripts/install-mingw.sh can't detect the builder's architecture"
            echo "please tweak it to recognize the builder named '${CI_JOB_NAME}'"
            exit 1
            ;;
    esac

    case "${mingw_archive}" in
        *.7z)
            curl -o mingw.7z "${MIRRORS_BASE}/${mingw_archive}"
            7z x -y mingw.7z > /dev/null
            ;;
        *.zip)
            curl -o mingw.zip "${MIRRORS_BASE}/${mingw_archive}"
            unzip -q mingw.zip
            mv llvm-mingw-20251104-ucrt-$arch $mingw_dir
            # Temporary workaround: https://github.com/mstorsjo/llvm-mingw/issues/493
            mkdir -p $mingw_dir/bin
            ln -s $arch-w64-windows-gnu.cfg $mingw_dir/bin/$arch-pc-windows-gnu.cfg
            ;;
        *)
            echo "Unrecognized archive type"
            exit 1
            ;;
    esac

    ciCommandAddPath "$(cygpath -m "$(pwd)/${mingw_dir}/bin")"
fi
