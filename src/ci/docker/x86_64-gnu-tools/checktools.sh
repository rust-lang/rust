#!/bin/sh

# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -eu

X_PY="$1"
TOOLSTATE_FILE="$(realpath $2)"
OS="$3"
COMMIT="$(git rev-parse HEAD)"

touch "$TOOLSTATE_FILE"

set +e
python2.7 "$X_PY" test --no-fail-fast \
    src/tools/rls \
    src/tools/rustfmt \
    src/tools/miri \
    src/tools/clippy
set -e

cat "$TOOLSTATE_FILE"

if [ "$RUST_RELEASE_CHANNEL" = nightly ]; then
    . "$(dirname $0)/repo.sh"
    MESSAGE_FILE=$(mktemp -t msg.XXXXXX)
    echo "($OS CI update)" > "$MESSAGE_FILE"
    commit_toolstate_change "$MESSAGE_FILE" \
        sed -i "1 a\\
$COMMIT\t$(cat "$TOOLSTATE_FILE")
" "history/$OS.tsv"
    rm -f "$MESSAGE_FILE"
    exit 0
fi

if grep -q fail "$TOOLSTATE_FILE"; then
    exit 4
fi
