#!/bin/bash
# Check out all our submodules, but more quickly than using git by using one of
# our custom scripts

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    path="/c/cache/rustsrc"
else
    path="${HOME}/rustsrc"
fi

mkdir -p "${path}"
"$(cd "$(dirname "$0")" && pwd)/../init_repo.sh" . "${path}"
