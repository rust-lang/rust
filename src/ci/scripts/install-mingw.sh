#!/bin/bash
# For mingw builds use a vendored mingw.

set -x

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

MINGW_ARCHIVE_32="i686-14.2.0-release-posix-dwarf-msvcrt-rt_v12-rev2.7z"
MINGW_ARCHIVE_64="x86_64-14.2.0-release-posix-seh-msvcrt-rt_v12-rev2.7z"
LLVM_MINGW_ARCHIVE_AARCH64="llvm-mingw-20251104-ucrt-aarch64.zip"
LLVM_MINGW_ARCHIVE_X86_64="llvm-mingw-20251104-ucrt-x86_64.zip"

if isWindows && isKnownToBeMingwBuild; then
    toolchains=
    # i686-pc-windows-gnu is cross-compiled from x86_64-pc-windows-gnu, so we need
    # the both toolchains
    if [[ "${CI_JOB_NAME}" == *i686-mingw* ]]; then
        toolchains=("${CI_JOB_NAME}" "x86_64-mingw")
    else
        toolchains=("${CI_JOB_NAME}")
    fi

    for toolchain in "${toolchains[@]}"; do
        case "${toolchain}" in
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

        # Stop /msys64/bin from being prepended to PATH by adding the bin directory manually.
        # Note that this intentionally uses a Windows style path instead of the msys2 path to
        # avoid being auto-translated into `/usr/bin`, which will not have the desired effect.
        msys2Path="c:/msys64"
        ciCommandAddPath "${msys2Path}/usr/bin"

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

        bindir="$(pwd)/${mingw_dir}/bin"
        ciCommandAddPath "$(cygpath -m "$bindir")"

        # FIXME(gcc16): We test with GCC15, which has f16 symbols incompatible with the
        # ABI used by LLVM and GCC16+. Hack around this by deleting the symbols from
        # `libgcc`, meaning our symbols from `compiler-builtins` will always be picked.
        if [[ "${toolchain}" = *"x86_64"* ]]; then
            tmp="$(mktemp -d)"
            trap 'rm -rf "$tmp"' EXIT
            cd "$tmp"

            libgcc="$("$bindir/gcc" -print-libgcc-file-name)"
            ar -x "$libgcc"

            for f in *.o; do
                echo "$f"
                "$bindir/objcopy" \
                    --strip-unneeded-symbol=__extendhfsf2 \
                    --strip-unneeded-symbol=__extendhfdf2 \
                    --strip-unneeded-symbol=__extendhftf2 \
                    --strip-unneeded-symbol=__truncsfhf2 \
                    --strip-unneeded-symbol=__truncdfhf2 \
                    --strip-unneeded-symbol=__trunctfhf2 \
                    --strip-unneeded-symbol=__fixhfsi \
                    --strip-unneeded-symbol=__fixhfdi \
                    --strip-unneeded-symbol=__fixhfti \
                    --strip-unneeded-symbol=__fixunshfsi \
                    --strip-unneeded-symbol=__fixunshfdi \
                    --strip-unneeded-symbol=__fixunshfti \
                    --strip-unneeded-symbol=__floatsihf \
                    --strip-unneeded-symbol=__floatdihf \
                    --strip-unneeded-symbol=__floattihf \
                    --strip-unneeded-symbol=__floatunsihf \
                    --strip-unneeded-symbol=__floatundihf \
                    --strip-unneeded-symbol=__floatuntihf \
                    "$f"
            done

            ar -r "$libgcc" *
        fi

        # Initialize mingw for the user.
        # This should be done by github but isn't for some reason.
        # (see https://github.com/actions/runner-images/issues/12600)
        /c/msys64/usr/bin/bash -lc ' '
    done
fi
