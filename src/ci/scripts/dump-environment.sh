#!/bin/bash
# This script dumps information about the build environment to stdout.

set -euo pipefail
IFS=$'\n\t'

echo "environment variables:"
printenv | sort
echo

echo "disk usage:"
df -h
echo

echo "biggest files in the working dir:"
set +o pipefail
du . | sort -nr | head -n100
set -o pipefail
echo
