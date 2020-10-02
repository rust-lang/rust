#!/bin/bash
# We've had multiple issues with the default disk running out of disk space
# during builds, and it looks like other disks mounted in the VMs have more
# space available. This script synlinks the build directory to those other
# disks, in the CI providers and OSes affected by this.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows && isAzurePipelines; then
    cmd //c "mkdir c:\\MORE_SPACE"
    cmd //c "mklink /J build c:\\MORE_SPACE"
fi
