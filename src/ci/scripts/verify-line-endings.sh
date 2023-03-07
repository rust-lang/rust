#!/bin/bash
# See also the disable for autocrlf, this just checks that it worked.
#
# We check both in rust-lang/rust and in a submodule to make sure both are
# accurate. Submodules are checked out significantly later than the main
# repository in this script, so settings can (and do!) change between then.
#
# Linux (and maybe macOS) builders don't currently have dos2unix so just only
# run this step on Windows.

set -euo pipefail
IFS=$'\n\t'

source "$(cd "$(dirname "$0")" && pwd)/../shared.sh"

if isWindows; then
    # print out the git configuration so we can better investigate failures in
    # the following
    git config --list --show-origin
    dos2unix -ih Cargo.lock src/tools/rust-installer/install-template.sh
    endings=$(dos2unix -ic Cargo.lock src/tools/rust-installer/install-template.sh)
    # if endings has non-zero length, error out
    if [ -n "$endings" ]; then exit 1 ; fi
fi
