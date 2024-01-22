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

    # Delete these pre-installed tools because we are using the MSYS2 setup action versions
    # instead, so we can't accidentally use them.
    # Delete Windows-Git
    rm -r "/c/Program Files/Git/"
    # Delete pre-installed version of MSYS2
    rm -r "/c/msys64/"
    # Delete Strawberry Perl, which contains a version of mingw
    rm -r "/c/Strawberry/"
    # Delete native CMake
    rm -r "/c/Program Files/CMake/"
    # Delete these other copies of mingw, I don't even know where they come from.
    rm -r "/c/mingw64/"
    rm -r "/c/mingw32/"
fi
