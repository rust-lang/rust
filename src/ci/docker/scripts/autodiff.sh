#!/bin/bash

set -ex

../x.py build --stage 1 library

test_status=0

for test in \
    tests/codegen-llvm/autodiff \
    tests/pretty/autodiff \
    tests/ui/autodiff \
    tests/run-make/autodiff \
    tests/ui/feature-gates/feature-gate-autodiff.rs
do
    set +e
    ../x.py test --stage 1 "$test"
    rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        continue
    fi

    echo "test failed: $test (exit status $rc)"

    # compiletest exits 1 for ordinary test failures; keep other statuses distinct.
    if [ "$rc" -ne 1 ]; then
        exit "$rc"
    fi

    test_status=1
done

exit "$test_status"
