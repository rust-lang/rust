#!/bin/bash
# Clean up and prepare the MSYS2 installation. MSYS2 is needed primarily for
# the test suite (run-make), but is also used by the MinGW toolchain for assembling things.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"
if isWindows; then
    # Detect the native Python version installed on the agent. On GitHub
    # Actions, the C:\hostedtoolcache\windows\Python directory contains a
    # subdirectory for each installed Python version.
    #
    # The -V flag of the sort command sorts the input by version number.
    native_python_version="$(ls /c/hostedtoolcache/windows/Python | sort -Vr | head -n 1)"

    # Make sure we use the native python interpreter instead of some msys equivalent
    # one way or another. The msys interpreters seem to have weird path conversions
    # baked in which break LLVM's build system one way or another, so let's use the
    # native version which keeps everything as native as possible.
    python_home="/c/hostedtoolcache/windows/Python/${native_python_version}/x64"
    if ! [[ -f "${python_home}/python3.exe" ]]; then
        cp "${python_home}/python.exe" "${python_home}/python3.exe"
    fi
    ciCommandAddPath "C:\\hostedtoolcache\\windows\\Python\\${native_python_version}\\x64"
    ciCommandAddPath "C:\\hostedtoolcache\\windows\\Python\\${native_python_version}\\x64\\Scripts"

    # Install pacboy for easily installing packages
    pacman -S --noconfirm pactoys

    # Remove these pre-installed tools so we can't accidentally use them, because we are using the
    # MSYS2 setup action versions instead. Because `rm -r`-ing them is slow, we mv them off path
    # instead.
    # Remove pre-installed version of MSYS2
    echo "Cleaning up existing tools in PATH"
    notpath="/c/NOT/ON/PATH/"
    mkdir --parents "$notpath"
    mv -t "$notpath" "/c/msys64/"
    # Remove Strawberry Perl, which contains a version of mingw
    mv -t "$notpath" "/c/Strawberry/"
    # Remove these other copies of mingw, I don't even know where they come from.
    mv -t "$notpath" "/c/mingw64/"
    mv -t "$notpath" "/c/mingw32/"
    echo "Finished cleaning up tools in PATH"

    if isKnownToBeMingwBuild; then
        # Use the mingw version of CMake for mingw builds.
        # However, the MSVC build needs native CMake, as it fails with the mingw one.
        # Delete native CMake
        rm -r "/c/Program Files/CMake/"
        # Install mingw-w64-$arch-cmake
        pacboy -S --noconfirm cmake:p

        # It would be nice to use MSYS's git in MinGW builds so that it's tested and known to
        # work. But it makes everything extremely slow, so it's commented out for now.
        # # Delete Windows-Git
        # rm -r "/c/Program Files/Git/"
        # # Install MSYS2 git
        # pacman -S --noconfirm git
    fi
fi
