#!/bin/bash
# Download and install MSYS2, needed primarily for the test suite (run-make) but
# also used by the MinGW toolchain for assembling things.
#
# FIXME: we should probe the default azure image and see if we can use the MSYS2
# toolchain there. (if there's even one there). For now though this gets the job
# done.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    # Pre-followed the api/v2 URL to the CDN since the API can be a bit flakey
    curl -sSL https://packages.chocolatey.org/msys2.20190524.0.0.20191030.nupkg > \
        msys2.nupkg
    curl -sSL https://packages.chocolatey.org/chocolatey-core.extension.1.3.5.1.nupkg > \
        chocolatey-core.extension.nupkg
    choco install -s . msys2 \
        --params="/InstallDir:$(ciCheckoutPath)/msys2 /NoPath" -y --no-progress
    rm msys2.nupkg chocolatey-core.extension.nupkg
    mkdir -p "$(ciCheckoutPath)/msys2/home/${USERNAME}"
    ciCommandAddPath "$(ciCheckoutPath)/msys2/usr/bin"

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
    cp "${python_home}/python.exe" "${python_home}/python3.exe"
    ciCommandAddPath "C:\\hostedtoolcache\\windows\\Python\\${native_python_version}\\x64"
    ciCommandAddPath "C:\\hostedtoolcache\\windows\\Python\\${native_python_version}\\x64\\Scripts"
fi
