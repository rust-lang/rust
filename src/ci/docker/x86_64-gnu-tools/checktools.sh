#!/bin/sh

set -eu

X_PY="$1"

# Try to test all the tools and store the build/test success in the TOOLSTATE_FILE

set +e
python2.7 "$X_PY" test --no-fail-fast \
    src/doc/book \
    src/doc/nomicon \
    src/doc/reference \
    src/doc/rust-by-example \
    src/doc/embedded-book \
    src/doc/edition-guide \
    src/doc/rustc-guide \
    src/tools/clippy \
    src/tools/rls \
    src/tools/rustfmt \
    src/tools/miri \

set -e

python2.7 "$X_PY" test check-tools
