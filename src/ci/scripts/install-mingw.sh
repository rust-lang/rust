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
    # echo "Path of / : $(cygpath -w /)"
    # echo "PATH: $PATH"
    # echo "MAJAHA PWD: $(pwd) | $(cygpath -w $(pwd))"
    # echo "MSYSTEM: ${MSYSTEM-unset}"
    # echo "MAJAHA 3: $(cygpath -w $(which git))"
    # echo "MAJAHA 3: $(cygpath -w $(which python))"
    # echo "GITHUB_PATH: $GITHUB_PATH"
    # cat "$GITHUB_PATH"
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
            # aarch64 is a cross-compiled target. Use the x86_64 #NOTE check msystem variable CI
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
        echo "1pacman -Qg mingw-w64-x86_64-toolchain:"
        # pacman -Qg mingw-w64-x86_64-toolchain # this gets run even for msvc jobs,
        # # checks if the package (group) is installed
        # pacman -Syu --noconfirm
        # pacman -S --noconfirm --needed mingw-w64-$arch-toolchain mingw-w64-$arch-cmake \
        #     mingw-w64-$arch-gcc \
        #     mingw-w64-$arch-python
        # ^the python package is actually for python3 #suspect, is this python even used?
        # #ciCommandAddPath "/mingw${bits}/bin"
        # ^alternatively, could maybe run bash without --noprofile and --norc in ci.yml
        # echo "CUSTOM MINGW PATH 0: /mingw${bits}/bin | $(cygpath -w "/mingw${bits}/bin")"
        # echo "2pacman -Qg mingw-w64-x86_64-toolchain:"
        # pacman -Qg mingw-w64-x86_64-toolchain
    else
        mingw_dir="mingw${bits}"

        curl -o mingw.7z "${MIRRORS_BASE}/${mingw_archive}"
        # ^This doesn't seem to include python. Should install in msys2 step instead?
        7z x -y mingw.7z > /dev/null
        ciCommandAddPath "$(pwd)/${mingw_dir}/bin"
        #echo "CUSTM MINGW PATH 1: $(pwd)/${mingw_dir}/bin | $(cygpath -w $(pwd)/${mingw_dir}/bin)"
    fi
    # echo "MAJAHA 4: $(cygpath -w $(which git))"
    # echo "MAJAHA 4: $(cygpath -w $(which python))"
    # echo "MAJAHA 4: $(cygpath -w $(which gcc))"
    echo "MAJAHA cmake: $(cygpath -w $(which cmake))"
    # echo "LS: $(ls)"
    # echo "GITHUB_PATH: $GITHUB_PATH"
    # cat "$GITHUB_PATH"
    # echo "MAJAHA /etc/pacman.conf"
    # cat /etc/pacman.conf
    # echo "\n"
    # echo "MAJAHA /etc/pacman.d/mirrorlist.mingw64"
    # cat /etc/pacman.d/mirrorlist.mingw64
    echo WHICH GCC:
    which gcc || true
    echo WHICH clang:
    which clang || true
    echo "#### LS OF /mingw$bits/bin/: ####"
    ls /mingw$bits/bin/
    echo "#### LS OF /clang$bits/bin/: ####"
    ls /clang$bits/bin/
fi
