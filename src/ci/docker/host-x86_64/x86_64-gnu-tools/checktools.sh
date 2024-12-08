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

# Testing Miri is a bit more complicated.
# We set the GC interval to the shortest possible value (0 would be off) to increase the chance
# that bugs which only surface when the GC runs at a specific time are more likely to cause CI to fail.
# This significantly increases the runtime of our test suite, or we'd do this in PR CI too.
if [ -z "${PR_CI_JOB:-}" ]; then
    MIRIFLAGS=-Zmiri-provenance-gc=1 python3 "$X_PY" test --stage 2 src/tools/miri src/tools/miri/cargo-miri
else
    python3 "$X_PY" test --stage 2 src/tools/miri src/tools/miri/cargo-miri
fi
# We natively run this script on x86_64-unknown-linux-gnu and x86_64-pc-windows-msvc.
# Also cover some other targets via cross-testing, in particular all tier 1 targets.
case $HOST_TARGET in
  x86_64-unknown-linux-gnu)
    # Only this branch runs in PR CI.
    # Fully test all main OSes, including a 32bit target.
    python3 "$X_PY" test --stage 2 src/tools/miri src/tools/miri/cargo-miri --target x86_64-apple-darwin
    python3 "$X_PY" test --stage 2 src/tools/miri src/tools/miri/cargo-miri --target i686-pc-windows-msvc
    # Only run "pass" tests for the remaining targets, which is quite a bit faster.
    python3 "$X_PY" test --stage 2 src/tools/miri --target x86_64-pc-windows-gnu --test-args pass
    python3 "$X_PY" test --stage 2 src/tools/miri --target i686-unknown-linux-gnu --test-args pass
    python3 "$X_PY" test --stage 2 src/tools/miri --target aarch64-unknown-linux-gnu --test-args pass
    python3 "$X_PY" test --stage 2 src/tools/miri --target s390x-unknown-linux-gnu --test-args pass
    ;;
  x86_64-pc-windows-msvc)
    # Strangely, Linux targets do not work here. cargo always says
    # "error: cannot produce cdylib for ... as the target ... does not support these crate types".
    # Only run "pass" tests, which is quite a bit faster.
    #FIXME: Re-enable this once CI issues are fixed
    # See <https://github.com/rust-lang/rust/issues/127883>
    # For now, these tests are moved to `x86_64-msvc-ext2` in `src/ci/github-actions/jobs.yml`.
    #python3 "$X_PY" test --stage 2 src/tools/miri --target aarch64-apple-darwin --test-args pass
    #python3 "$X_PY" test --stage 2 src/tools/miri --target i686-pc-windows-gnu --test-args pass
    ;;
  *)
    echo "FATAL: unexpected host $HOST_TARGET"
    exit 1
    ;;
esac
# Also smoke-test `x.py miri`. This doesn't run any actual tests (that would take too long),
# but it ensures that the crates build properly when tested with Miri.

#FIXME: Re-enable this for msvc once CI issues are fixed
if [ "$HOST_TARGET" != "x86_64-pc-windows-msvc" ]; then
  python3 "$X_PY" miri --stage 2 library/core --test-args notest
  python3 "$X_PY" miri --stage 2 library/alloc --test-args notest
  python3 "$X_PY" miri --stage 2 library/std --test-args notest
fi
