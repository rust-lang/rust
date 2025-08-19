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
    --features "build-mpfr,libm/unstable,libm/force-soft-floats"
    --profile release-checked
)

# Run the non-extensive tests first to catch any easy failures
"${test_cmd[@]}" -- "$TO_TEST"

LIBM_EXTENSIVE_TESTS="$TO_TEST" "${test_cmd[@]}" -- extensive
