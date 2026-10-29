#!/bin/bash

set -euo pipefail

echo "Tests to run: '$TO_TEST'"

if [ -z "$TO_TEST" ]; then
    echo "No tests to run, exiting."
    exit
fi

set -x

test_cmd=(
    cargo test
    --package libm-test
    --no-default-features
    # Don't enable `arch` for extensive tests. Usually anything in asm is
    # only a single instruction or a small sequence, and we rely on the
    # vendors to test that for us.
    #
    # libm/unstable enables libm/unstable-intrinsics, which means we usually
    # get the single-instruction ops anyway when we aren't specifically
    # testing for them.
    --features "libm-test/build-mpfr libm-test/unstable-float libm/unstable"
    --profile release-checked
)

# Run the non-extensive tests first to catch any easy failures
"${test_cmd[@]}" -- "$TO_TEST"

LIBM_EXTENSIVE_TESTS="$TO_TEST" "${test_cmd[@]}" -- extensive
