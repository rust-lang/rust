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
CHANGED_FILES="$(git diff --name-status HEAD HEAD^)"
SIX_WEEK_CYCLE="$(( ($(date +%s) / 604800 - 3) % 6 ))"
# ^ 1970 Jan 1st is a Thursday, and our release dates are also on Thursdays,
#   thus we could divide by 604800 (7 days in seconds) directly.

touch "$TOOLSTATE_FILE"

# Try to test all the tools and store the build/test success in the TOOLSTATE_FILE

set +e
python2.7 "$X_PY" test --no-fail-fast \
    src/doc/book \
    src/doc/nomicon \
    src/doc/reference \
    src/doc/rust-by-example \
    src/tools/rls \
    src/tools/rustfmt \
    src/tools/miri \
    src/tools/clippy
set -e

cat "$TOOLSTATE_FILE"
echo

# This function checks that if a tool's submodule changed, the tool's state must improve
verify_status() {
    echo "Verifying status of $1..."
    if echo "$CHANGED_FILES" | grep -q "^M[[:blank:]]$2$"; then
        echo "This PR updated '$2', verifying if status is 'test-pass'..."
        if grep -vq '"'"$1"'":"test-pass"' "$TOOLSTATE_FILE"; then
            echo
            echo "⚠️ We detected that this PR updated '$1', but its tests failed."
            echo
            echo "If you do intend to update '$1', please check the error messages above and"
            echo "commit another update."
            echo
            echo "If you do NOT intend to update '$1', please ensure you did not accidentally"
            echo "change the submodule at '$2'. You may ask your reviewer for the"
            echo "proper steps."
            exit 3
        fi
    fi
}

# deduplicates the submodule check and the assertion that on beta some tools MUST be passing
check_dispatch() {
    if [ "$1" = submodule_changed ]; then
        # ignore $2 (branch id)
        verify_status $3 $4
    elif [ "$2" = beta ]; then
        echo "Requiring test passing for $3..."
        if grep -q '"'"$3"'":"\(test\|build\)-fail"' "$TOOLSTATE_FILE"; then
            exit 4
        fi
    fi
}

# list all tools here
status_check() {
    check_dispatch $1 beta book src/doc/book
    check_dispatch $1 beta nomicon src/doc/nomicon
    check_dispatch $1 beta reference src/doc/reference
    check_dispatch $1 beta rust-by-example src/doc/rust-by-example
    check_dispatch $1 beta rls src/tool/rls
    check_dispatch $1 beta rustfmt src/tool/rustfmt
    # these tools are not required for beta to successfully branch
    check_dispatch $1 nightly clippy-driver src/tool/clippy
    check_dispatch $1 nightly miri src/tool/miri
}

# If this PR is intended to update one of these tools, do not let the build pass
# when they do not test-pass.

status_check "submodule_changed"

CHECK_NOT="$(readlink -f "$(dirname $0)/checkregression.py")"
change_toolstate() {
    # only update the history
    if python2.7 "$CHECK_NOT" "$OS" "$TOOLSTATE_FILE" "_data/latest.json" changed; then
        echo 'Toolstate is not changed. Not updating.'
    else
        if [ $SIX_WEEK_CYCLE -eq 5 ]; then
            python2.7 "$CHECK_NOT" "$OS" "$TOOLSTATE_FILE" "_data/latest.json" regressed
        fi
        sed -i "1 a\\
$COMMIT\t$(cat "$TOOLSTATE_FILE")
" "history/$OS.tsv"
    fi
}

if [ "$RUST_RELEASE_CHANNEL" = nightly -a -n "${TOOLSTATE_REPO_ACCESS_TOKEN+is_set}" ]; then
    . "$(dirname $0)/repo.sh"
    MESSAGE_FILE=$(mktemp -t msg.XXXXXX)
    echo "($OS CI update)" > "$MESSAGE_FILE"
    commit_toolstate_change "$MESSAGE_FILE" change_toolstate
    rm -f "$MESSAGE_FILE"
    exit 0
fi

# abort compilation if an important tool doesn't build
# (this code is reachable if not on the nightly channel)
status_check "beta_required"
