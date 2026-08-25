#!/bin/bash
# See also the disable for autocrlf, this just checks that it worked.
#
# We check both in rust-lang/rust and in a submodule to make sure both are
# accurate. Submodules are checked out significantly later than the main
# repository in this script, so settings can (and do!) change between then.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

# print out the git configuration so we can better investigate failures in
# the following
git config --list --show-origin
# -U is necessary on Windows to stop grep automatically converting the line ending
endings=$(grep -Ul $(printf '\r$') Cargo.lock src/tools/cargo/Cargo.lock) || true
# if endings has non-zero length, error out
if [[ -n $endings ]]; then
    echo "Error: found DOS line endings"
    # Print the files with DOS line endings
    echo "$endings"
    exit 1
fi
