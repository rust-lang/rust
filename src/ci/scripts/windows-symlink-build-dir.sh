#!/bin/bash
# We've had issues with the default drive in use running out of space during a
# build, and it looks like the `C:` drive has more space than the default `D:`
# drive. We should probably confirm this with the azure pipelines team at some
# point, but this seems to fix our "disk space full" problems.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    cmd //c "mkdir c:\\MORE_SPACE"
    cmd //c "mklink /J build c:\\MORE_SPACE"
fi
