#!/bin/bash
# Clean up and prepare the MSYS2 installation.
# MSYS2 is used by the MinGW toolchain for assembling things.

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
fi
