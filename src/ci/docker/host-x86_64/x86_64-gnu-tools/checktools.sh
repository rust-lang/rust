#!/bin/sh
# ignore-tidy-linelength

set -eu
set -x # so one can see where we are in the script

X_PY="$1"

# Try to test the toolstate-tracked tools and store the build/test success in the TOOLSTATE_FILE.

# Pre-build the compiler and the library first to output a better error message when the build
# itself fails (see https://github.com/rust-lang/rust/issues/127869 for context).
python3 "$X_PY" build --stage 2 compiler rustdoc

set +e
python3 "$X_PY" test --stage 2 --no-fail-fast \
    src/doc/book \
    src/doc/nomicon \
    src/doc/reference \
    src/doc/rust-by-example \
    src/doc/embedded-book \
    src/doc/edition-guide \

set -e

# debugging: print out the saved toolstates
cat /tmp/toolstate/toolstates.json

# Test remaining tools that must pass.
python3 "$X_PY" test --stage 2 check-tools
python3 "$X_PY" test --stage 2 src/tools/clippy
python3 "$X_PY" test --stage 2 src/tools/rustfmt

# The below is a regression test for https://github.com/rust-lang/rust/pull/146501#issuecomment-3292608398.
# The bug caused 0 tests to run. By grepping on that 1 test is run we prevent regressing.
# Any test can be used. We arbitrarily chose `tests/ui/lint/unused/unused-result.rs`.
python3 "$X_PY" test tests/ui --test-args tests/ui/lint/unused/unused-result.rs --force-rerun |
grep --fixed-strings 'test result: ok. 1 passed; 0 failed; 0 ignored;' ||
( echo "ERROR: --test-args functionality is broken" && exit 1 )
