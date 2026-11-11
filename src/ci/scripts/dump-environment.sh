#!/bin/bash
# This script dumps information about the build environment to stdout.

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

set -euo pipefail
IFS=$'\n\t'

echo "environment variables:"
printenv | sort
echo

echo "disk usage:"
df -h
echo

echo "biggest files in the working dir:"
du . | sort -n | tail -n100 | sort -nr # because piping sort to head gives a broken pipe
echo

if isMacOS
then
    # Debugging information that might be helpful for diagnosing macOS
    # performance issues.
    # SIP
    csrutil status
    # Gatekeeper
    spctl --status
    # Authorization policy
    DevToolsSecurity -status
    # Spotlight status
    mdutil -avs
fi
