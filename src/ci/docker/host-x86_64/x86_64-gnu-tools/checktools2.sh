#!/bin/sh
# ignore-tidy-linelength

set -eu
set -x # so one can see where we are in the script

X_PY="$1"

# python3 "$X_PY" --stage 2 test src/tools/cargo -- --no-run
# python3 "$X_PY" --stage 2 build src/tools/cargo

# Try to test the toolstate-tracked tools and store the build/test success in the TOOLSTATE_FILE.

# Pre-build the compiler and the library first to output a better error message when the build
# itself fails (see https://github.com/rust-lang/rust/issues/127869 for context).
# python3 "$X_PY" build --stage 2 compiler rustdoc

# set +e
# python3 "$X_PY" test --stage 2 --no-fail-fast \
#     src/doc/book \
#     src/doc/nomicon \
#     src/doc/reference \
#     src/doc/rust-by-example \
#     src/doc/embedded-book \
#     src/doc/edition-guide \

# set -e

# debugging: print out the saved toolstates
# cat /tmp/toolstate/toolstates.json

# Test remaining tools that must pass.
# python3 "$X_PY" test --stage 2 check-tools
# python3 "$X_PY" test --stage 2 src/tools/clippy
# python3 "$X_PY" test --stage 2 src/tools/rustfmt

python3 "$X_PY" test --stage 2 src/tools/miri src/tools/miri/cargo-miri
python3 "$X_PY" test --stage 2 src/tools/miri --target aarch64-apple-darwin --test-args pass
python3 "$X_PY" test --stage 2 src/tools/miri --target i686-pc-windows-gnu --test-args pass

# Also smoke-test `x.py miri`. This doesn't run any actual tests (that would take too long),
# but it ensures that the crates build properly when tested with Miri.
# python3 "$X_PY" miri --stage 2 library/core --test-args notest
# python3 "$X_PY" miri --stage 2 library/alloc --test-args notest
# python3 "$X_PY" miri --stage 2 library/std --test-args notest
