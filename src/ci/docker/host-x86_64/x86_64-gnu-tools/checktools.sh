#!/bin/sh

set -eu

X_PY="$1"

# Try to test all the tools and store the build/test success in the TOOLSTATE_FILE

set +e
python3 "$X_PY" test --stage 2 --no-fail-fast \
    src/doc/book \
    src/doc/nomicon \
    src/doc/reference \
    src/doc/rust-by-example \
    src/doc/embedded-book \
    src/doc/edition-guide \
    src/tools/miri \

set -e

# debugging: print out the saved toolstates
cat /tmp/toolstate/toolstates.json
python3 "$X_PY" test --stage 2 check-tools
python3 "$X_PY" test --stage 2 src/tools/clippy
python3 "$X_PY" test --stage 2 src/tools/rustfmt
