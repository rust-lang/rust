#!/bin/sh

set -eu

X_PY="$1"
TOOLSTATE_FILE="$(realpath $2)"
OS="$3"
COMMIT="$(git rev-parse HEAD)"
CHANGED_FILES="$(git diff --name-status HEAD HEAD^)"
SIX_WEEK_CYCLE="$(( ($(date +%s) / 86400 - 20) % 42 ))"
# ^ Number of days after the last promotion of beta.
#   Its value is 41 on the Tuesday where "Promote master to beta (T-2)" happens.
#   The Wednesday after this has value 0.
#   We track this value to prevent regressing tools in the last week of the 6-week cycle.

touch "$TOOLSTATE_FILE"

# Try to test all the tools and store the build/test success in the TOOLSTATE_FILE

set +e
python2.7 "$X_PY" test --no-fail-fast \
    src/doc/book \
    src/doc/nomicon \
    src/doc/reference \
    src/doc/rust-by-example \
    src/doc/embedded-book \
    src/doc/edition-guide \
    src/tools/clippy \
    src/tools/rls \
    src/tools/rustfmt \
    src/tools/miri \

set -e

cat "$TOOLSTATE_FILE"
echo

# This function checks if a particular tool is *not* in status "test-pass".
check_tool_failed() {
    grep -vq '"'"$1"'":"test-pass"' "$TOOLSTATE_FILE"
}

# This function checks that if a tool's submodule changed, the tool's state must improve
verify_status() {
    echo "Verifying status of $1..."
    if echo "$CHANGED_FILES" | grep -q "^M[[:blank:]]$2$"; then
        echo "This PR updated '$2', verifying if status is 'test-pass'..."
        if check_tool_failed "$1"; then
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

# deduplicates the submodule check and the assertion that on beta some tools MUST be passing.
# $1 should be "submodule_changed" to only check tools that got changed by this PR,
# or "beta_required" to check all tools that have $2 set to "beta".
check_dispatch() {
    if [ "$1" = submodule_changed ]; then
        # ignore $2 (branch id)
        verify_status $3 $4
    elif [ "$2" = beta ]; then
        echo "Requiring test passing for $3..."
        if check_tool_failed "$3"; then
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
    check_dispatch $1 beta edition-guide src/doc/edition-guide
    check_dispatch $1 beta rls src/tools/rls
    check_dispatch $1 beta rustfmt src/tools/rustfmt
    check_dispatch $1 beta clippy-driver src/tools/clippy
    # these tools are not required for beta to successfully branch
    check_dispatch $1 nightly miri src/tools/miri
    check_dispatch $1 nightly embedded-book src/doc/embedded-book
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
        if [ $SIX_WEEK_CYCLE -ge 35 ]; then
            python2.7 "$CHECK_NOT" "$OS" "$TOOLSTATE_FILE" "_data/latest.json" regressed
        fi
        sed -i "1 a\\
$COMMIT\t$(cat "$TOOLSTATE_FILE")
" "history/$OS.tsv"
    fi
}

if [ "$RUST_RELEASE_CHANNEL" = nightly ]; then
    if [ -n "${TOOLSTATE_REPO_ACCESS_TOKEN+is_set}" ]; then
        . "$(dirname $0)/repo.sh"
        MESSAGE_FILE=$(mktemp -t msg.XXXXXX)
        echo "($OS CI update)" > "$MESSAGE_FILE"
        commit_toolstate_change "$MESSAGE_FILE" change_toolstate
        rm -f "$MESSAGE_FILE"
    fi
    exit 0
fi

# abort compilation if an important tool doesn't build
# (this code is reachable if not on the nightly channel)
status_check "beta_required"
