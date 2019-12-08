#!/bin/bash
# Spawn a background process to collect CPU usage statistics which we'll upload
# at the end of the build. See the comments in the script here for more
# information.

set -euo pipefail
IFS=$'\n\t'

python src/ci/cpu-usage-over-time.py &> cpu-usage.csv &
