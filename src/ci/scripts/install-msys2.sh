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
    # FIXME(mati865): remove `/NoUpdate` once MSYS2 issue is fixed
    choco install -s . msys2 \
        --params="/InstallDir:$(ciCheckoutPath)/msys2 /NoPath /NoUpdate" -y --no-progress
    rm msys2.nupkg chocolatey-core.extension.nupkg
    mkdir -p "$(ciCheckoutPath)/msys2/home/${USERNAME}"
    ciCommandAddPath "$(ciCheckoutPath)/msys2/usr/bin"

    echo "switching shell to use our own bash"
    ciCommandSetEnv CI_OVERRIDE_SHELL "$(ciCheckoutPath)/msys2/usr/bin/bash.exe"
fi
