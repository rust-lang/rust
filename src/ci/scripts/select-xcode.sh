#!/bin/bash
# This script selects the Xcode instance to use.
# It also tries to do some cleanup in CI jobs of unused Xcodes.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isMacOS; then
    # This additional step is to try to remove an Xcode we aren't using because each one is HUGE
    old_xcode="$(xcode-select --print-path)"
    old_xcode="${old_xcode%/*}" # pop a dir
    old_xcode="${old_xcode%/*}" # twice
    if [[ $old_xcode =~ $SELECT_XCODE ]]; then
        echo "xcode-select.sh's brutal hack may not be necessary?"
        exit 1
    elif [[ $SELECT_XCODE =~ "16" ]]; then
        echo "Using Xcode 16? Please fix xcode-select.sh"
        exit 1
    fi
    if [ $CI ]; then # just in case someone sources this on their real computer
        sudo rm -rf "${old_xcode}"
        xcode_16="${old_xcode%/*}/Xcode-16.0.0.app"
        sudo rm -rf "${xcode_16}"
    fi
    sudo xcode-select -s "${SELECT_XCODE}"
fi
