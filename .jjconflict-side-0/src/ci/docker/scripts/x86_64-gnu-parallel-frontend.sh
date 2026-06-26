#!/usr/bin/env bash

set -eux -o pipefail

if [ ! -v RUST_TEST_THREADS ]; then
    RUST_TEST_THREADS_=$(python3 -c "print(max(1, $(nproc) // ${PARALLEL_FRONTEND_THREADS}))")
else
    RUST_TEST_THREADS_=${RUST_TEST_THREADS}
fi

RUST_TEST_THREADS=${RUST_TEST_THREADS_} \
    python3 ../x.py --stage 2 test \
    tests/ui \
    -- \
    --parallel-frontend-threads="${PARALLEL_FRONTEND_THREADS}" \
    --iteration-count="${ITERATION_COUNT}"
