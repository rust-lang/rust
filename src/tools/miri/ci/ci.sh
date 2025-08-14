#!/bin/bash
set -eu

function begingroup {
  echo "::group::$@"
  set -x
}

function endgroup {
  set +x
  echo "::endgroup"
}

begingroup "Sanity-check environment"

# Ensure the HOST_TARGET is what it should be.
if ! rustc -vV | grep -q "^host: $HOST_TARGET\$"; then
  echo "This runner should be using host target $HOST_TARGET but rustc disagrees:"
  rustc -vV
  exit 1
fi

endgroup

begingroup "Building Miri"

# Global configuration
export RUSTFLAGS="-D warnings"
export CARGO_INCREMENTAL=0
export CARGO_EXTRA_FLAGS="--locked"

# Determine configuration for installed build (used by test-cargo-miri and `./miri bench`).
echo "Installing release version of Miri"
time ./miri install

# Prepare debug build for direct `./miri` invocations.
# We enable all features to make sure the Stacked Borrows consistency check runs.
echo "Building debug version of Miri"
export CARGO_EXTRA_FLAGS="$CARGO_EXTRA_FLAGS --all-features"
time ./miri build # the build that all the `./miri test` below will use

endgroup

# Run tests. Recognizes these variables:
# - TEST_TARGET: the target to test. Empty for host target.
# - GC_STRESS: if non-empty, run the GC stress test for the main test suite.
# - MIR_OPT: if non-empty, re-run test `pass` tests with mir-opt-level=4
# - MANY_SEEDS: if set to N, run the "many-seeds" tests N times
# - TEST_BENCH: if non-empty, check that the benchmarks all build
# - CARGO_MIRI_ENV: if non-empty, set some env vars and config to potentially confuse cargo-miri
function run_tests {
  if [ -n "${TEST_TARGET-}" ]; then
    begingroup "Testing foreign architecture $TEST_TARGET"
    TARGET_FLAG="--target $TEST_TARGET"
    MULTI_TARGET_FLAG=""
  else
    begingroup "Testing host architecture"
    TARGET_FLAG=""
    MULTI_TARGET_FLAG="--multi-target"
  fi

  ## ui test suite
  if [ -n "${GC_STRESS-}" ]; then
    time MIRIFLAGS="${MIRIFLAGS-} -Zmiri-provenance-gc=1" ./miri test $TARGET_FLAG
  else
    time ./miri test $TARGET_FLAG
  fi

  ## advanced tests
  if [ -n "${MIR_OPT-}" ]; then
    # Tests with optimizations (`-O` is what cargo passes, but crank MIR optimizations up all the
    # way, too).
    # Optimizations change diagnostics (mostly backtraces), so we don't check
    # them. Also error locations change so we don't run the failing tests.
    # We explicitly enable debug-assertions here, they are disabled by -O but we have tests
    # which exist to check that we panic on debug assertion failures.
    time MIRIFLAGS="${MIRIFLAGS-} -O -Zmir-opt-level=4 -Cdebug-assertions=yes" MIRI_SKIP_UI_CHECKS=1 ./miri test $TARGET_FLAG tests/{pass,panic}
  fi
  if [ -n "${MANY_SEEDS-}" ]; then
    # Run many-seeds tests. (Also tests `./miri run`.)
    time for FILE in tests/many-seeds/*.rs; do
      ./miri run "-Zmiri-many-seeds=0..$MANY_SEEDS" $TARGET_FLAG "$FILE"
    done
  fi
  if [ -n "${TEST_BENCH-}" ]; then
    # Check that the benchmarks build and run, but only once.
    time HYPERFINE="hyperfine -w0 -r1 --show-output" ./miri bench $TARGET_FLAG --no-install
  fi
  # Smoke-test `./miri run --dep`.
  ./miri run $TARGET_FLAG --dep tests/pass-dep/getrandom.rs

  ## test-cargo-miri
  # On Windows, there is always "python", not "python3" or "python2".
  if command -v python3 > /dev/null; then
    PYTHON=python3
  else
    PYTHON=python
  fi
  # Some environment setup that attempts to confuse the heck out of cargo-miri.
  if [ -n "${CARGO_MIRI_ENV-}" ]; then
    # These act up on Windows (`which miri` produces a filename that does not exist?!?).
    # RUSTC is the main thing to set (it changes the first argument our wrapper will see).
    # Unless MIRI is also set, that produces a warning.
    export RUSTC=$(which rustc)
    export MIRI=$(rustc +miri --print sysroot)/bin/miri
    # We entirely ignore other wrappers.
    mkdir -p .cargo
    echo 'build.rustc-wrapper = "thisdoesnotexist"' > .cargo/config.toml
  fi
  # Run the actual test
  time ${PYTHON} test-cargo-miri/run-test.py $TARGET_FLAG $MULTI_TARGET_FLAG
  # Clean up
  unset RUSTC MIRI
  rm -rf .cargo

  endgroup
}

function run_tests_minimal {
  if [ -n "${TEST_TARGET-}" ]; then
    begingroup "Testing MINIMAL foreign architecture $TEST_TARGET: only testing $@"
    TARGET_FLAG="--target $TEST_TARGET"
  else
    echo "run_tests_minimal requires TEST_TARGET to be set"
    exit 1
  fi

  time ./miri test $TARGET_FLAG "$@"

  # Ensure that a small smoke test of cargo-miri works.
  time cargo miri run --manifest-path test-cargo-miri/no-std-smoke/Cargo.toml $TARGET_FLAG

  endgroup
}

## Main Testing Logic ##

# In particular, fully cover all tier 1 targets.
# We also want to run the many-seeds tests on all tier 1 targets.
case $HOST_TARGET in
  x86_64-unknown-linux-gnu)
    # Host
    GC_STRESS=1 MIR_OPT=1 MANY_SEEDS=64 TEST_BENCH=1 CARGO_MIRI_ENV=1 run_tests
    # Extra tier 1
    MANY_SEEDS=64 TEST_TARGET=x86_64-apple-darwin run_tests
    MANY_SEEDS=64 TEST_TARGET=x86_64-pc-windows-gnu run_tests
    ;;
  i686-unknown-linux-gnu)
    # Host
    # Without GC_STRESS as this is a slow runner.
    MIR_OPT=1 MANY_SEEDS=64 TEST_BENCH=1 CARGO_MIRI_ENV=1 run_tests
    # Partially supported targets (tier 2)
    BASIC="empty_main integer heap_alloc libc-mem vec string btreemap" # ensures we have the basics: pre-main code, system allocator
    UNIX="hello panic/panic panic/unwind concurrency/simple atomic libc-mem libc-misc libc-random env num_cpus" # the things that are very similar across all Unixes, and hence easily supported there
    TEST_TARGET=aarch64-linux-android  run_tests_minimal $BASIC $UNIX time hashmap random thread sync concurrency epoll eventfd
    TEST_TARGET=wasm32-wasip2          run_tests_minimal $BASIC wasm
    TEST_TARGET=wasm32-unknown-unknown run_tests_minimal no_std empty_main wasm # this target doesn't really have std
    TEST_TARGET=thumbv7em-none-eabihf  run_tests_minimal no_std
    ;;
  aarch64-unknown-linux-gnu)
    # Host
    GC_STRESS=1 MIR_OPT=1 MANY_SEEDS=64 TEST_BENCH=1 CARGO_MIRI_ENV=1 run_tests
    # Extra tier 2
    MANY_SEEDS=16 TEST_TARGET=arm-unknown-linux-gnueabi run_tests # 32bit ARM
    MANY_SEEDS=16 TEST_TARGET=aarch64-pc-windows-gnullvm run_tests # gnullvm ABI
    MANY_SEEDS=16 TEST_TARGET=s390x-unknown-linux-gnu run_tests # big-endian architecture of choice
    # Custom target JSON file
    TEST_TARGET=tests/x86_64-unknown-kernel.json MIRI_NO_STD=1 run_tests_minimal no_std
    ;;
  armv7-unknown-linux-gnueabihf)
    # Host
    GC_STRESS=1 MIR_OPT=1 MANY_SEEDS=64 TEST_BENCH=1 CARGO_MIRI_ENV=1 run_tests
    ;;
  aarch64-apple-darwin)
    # Host
    GC_STRESS=1 MIR_OPT=1 MANY_SEEDS=64 TEST_BENCH=1 CARGO_MIRI_ENV=1 run_tests
    # Extra tier 1
    MANY_SEEDS=64 TEST_TARGET=i686-pc-windows-gnu run_tests
    MANY_SEEDS=64 TEST_TARGET=x86_64-pc-windows-msvc CARGO_MIRI_ENV=1 run_tests
    # Not officially supported tier 2
    MANY_SEEDS=16 TEST_TARGET=mips-unknown-linux-gnu run_tests # a 32bit big-endian target, and also a target without 64bit atomics
    MANY_SEEDS=16 TEST_TARGET=x86_64-unknown-illumos run_tests
    MANY_SEEDS=16 TEST_TARGET=x86_64-pc-solaris run_tests
    MANY_SEEDS=16 TEST_TARGET=x86_64-unknown-freebsd run_tests
    MANY_SEEDS=16 TEST_TARGET=i686-unknown-freebsd run_tests
    ;;
  i686-pc-windows-msvc)
    # Host
    # Without GC_STRESS as this is a very slow runner.
    MIR_OPT=1 MANY_SEEDS=64 TEST_BENCH=1 run_tests
    # Extra tier 1
    # We really want to ensure a Linux target works on a Windows host,
    # and a 64bit target works on a 32bit host.
    TEST_TARGET=x86_64-unknown-linux-gnu run_tests
    ;;
  aarch64-pc-windows-msvc)
    # Host
    # Without GC_STRESS as this is a very slow runner.
    MIR_OPT=1 MANY_SEEDS=64 TEST_BENCH=1 CARGO_MIRI_ENV=1 run_tests
    # Extra tier 1
    MANY_SEEDS=64 TEST_TARGET=i686-unknown-linux-gnu run_tests
    ;;
  *)
    echo "FATAL: unknown host target: $HOST_TARGET"
    exit 1
    ;;
esac
