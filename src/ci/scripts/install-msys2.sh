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
    for RETRY_COUNT in 1 2 3 4 5 6 7 8 9 10; do
        choco install msys2 \
            --params="/InstallDir:$(ciCheckoutPath)/msys2 /NoPath" -y --no-progress \
            && mkdir -p "$(ciCheckoutPath)/msys2/home/${USERNAME}" \
            && ciCommandAddPath "$(ciCheckoutPath)/msys2/usr/bin" && break
    done
fi
