#!/bin/bash
# Switch to XCode 9.3 on OSX since it seems to be the last version that supports
# i686-apple-darwin. We'll eventually want to upgrade this and it will probably
# force us to drop i686-apple-darwin, but let's keep the wheels turning for now.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isMacOS; then
    sudo xcode-select --switch /Applications/Xcode_9.3.app
fi
