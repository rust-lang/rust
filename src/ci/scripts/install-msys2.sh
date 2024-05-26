#!/bin/bash
# Clean up and prepare the MSYS2 installation.
# MSYS2 is used by the MinGW toolchain for assembling things.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"
