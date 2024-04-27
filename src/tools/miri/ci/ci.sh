#!/bin/bash
set -euo pipefail

function begingroup {
  echo "::group::$@"
  set -x
}

function endgroup {
  set +x
  echo "::endgroup"
}

begingroup "Building Miri"

# Special Windows hacks
if [ "$HOST_TARGET" = i686-pc-windows-msvc ]; then
  # The $BASH variable is `/bin/bash` here, but that path does not actually work. There are some
  # hacks in place somewhere to try to paper over this, but the hacks dont work either (see
  # <https://github.com/rust-lang/miri/pull/3402>). So we hard-code the correct location for Github
  # CI instead.
  BASH="C:/Program Files/Git/usr/bin/bash"
fi

# Global configuration
export RUSTFLAGS="-D warnings"
export CARGO_INCREMENTAL=0
export CARGO_EXTRA_FLAGS="--locked"

# Determine configuration for installed build (used by test-cargo-miri).
echo "Installing release version of Miri"
time ./miri install

# Prepare debug build for direct `./miri` invocations.
# We enable all features to make sure the Stacked Borrows consistency check runs.
echo "Building debug version of Miri"
export CARGO_EXTRA_FLAGS="$CARGO_EXTRA_FLAGS --all-features"
time ./miri build --all-targets # the build that all the `./miri test` below will use

endgroup

# Run tests. Recognizes these variables:
# - MIRI_TEST_TARGET: the target to test. Empty for host target.
# - GC_STRESS: if non-empty, run the GC stress test for the main test suite.
# - MIR_OPT: if non-empty, re-run test `pass` tests with mir-opt-level=4
# - MANY_SEEDS: if set to N, run the "many-seeds" tests N times
# - TEST_BENCH: if non-empty, check that the benchmarks all build
# - CARGO_MIRI_ENV: if non-empty, set some env vars and config to potentially confuse cargo-miri
function run_tests {
  if [ -n "${MIRI_TEST_TARGET-}" ]; then
    begingroup "Testing foreign architecture $MIRI_TEST_TARGET"
  else
    begingroup "Testing host architecture"
  fi

  ## ui test suite
  if [ -n "${GC_STRESS-}" ]; then
    time MIRIFLAGS="${MIRIFLAGS-} -Zmiri-provenance-gc=1" ./miri test
  else
    time ./miri test
  fi

  ## advanced tests
  if [ -n "${MIR_OPT-}" ]; then
    # Tests with optimizations (`-O` is what cargo passes, but crank MIR optimizations up all the
    # way, too).
    # Optimizations change diagnostics (mostly backtraces), so we don't check
    # them. Also error locations change so we don't run the failing tests.
    # We explicitly enable debug-assertions here, they are disabled by -O but we have tests
    # which exist to check that we panic on debug assertion failures.
    time MIRIFLAGS="${MIRIFLAGS-} -O -Zmir-opt-level=4 -Cdebug-assertions=yes" MIRI_SKIP_UI_CHECKS=1 ./miri test -- tests/{pass,panic}
  fi
  if [ -n "${MANY_SEEDS-}" ]; then
    # Also run some many-seeds tests. 64 seeds means this takes around a minute per test.
    # (Need to invoke via explicit `bash -c` for Windows.)
    time for FILE in tests/many-seeds/*.rs; do
      MIRI_SEEDS=$MANY_SEEDS ./miri many-seeds "$BASH" -c "./miri run '$FILE'"
    done
  fi
  if [ -n "${TEST_BENCH-}" ]; then
    # Check that the benchmarks build and run, but only once.
    time HYPERFINE="hyperfine -w0 -r1" ./miri bench
  fi

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
  time ${PYTHON} test-cargo-miri/run-test.py
  # Clean up
  unset RUSTC MIRI
  rm -rf .cargo

  endgroup
}

function run_tests_minimal {
  if [ -n "${MIRI_TEST_TARGET-}" ]; then
    begingroup "Testing MINIMAL foreign architecture $MIRI_TEST_TARGET: only testing $@"
  else
    echo "run_tests_minimal requires MIRI_TEST_TARGET to be set"
    exit 1
  fi

  ./miri test -- "$@"

  # Ensure that a small smoke test of cargo-miri works.
  cargo miri run --manifest-path test-cargo-miri/no-std-smoke/Cargo.toml --target ${MIRI_TEST_TARGET-$HOST_TARGET}

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
    # With reduced many-seed count to avoid spending too much time on that.
    # (All OSes are run with 64 seeds at least once though via the macOS runner.)
    MANY_SEEDS=16 MIRI_TEST_TARGET=i686-unknown-linux-gnu run_tests
    MANY_SEEDS=16 MIRI_TEST_TARGET=aarch64-unknown-linux-gnu run_tests
    MANY_SEEDS=16 MIRI_TEST_TARGET=x86_64-apple-darwin run_tests
    MANY_SEEDS=16 MIRI_TEST_TARGET=x86_64-pc-windows-gnu run_tests
    # Extra tier 2
    MIRI_TEST_TARGET=aarch64-apple-darwin run_tests
    MIRI_TEST_TARGET=arm-unknown-linux-gnueabi run_tests
    # Partially supported targets (tier 2)
    MIRI_TEST_TARGET=x86_64-unknown-freebsd run_tests_minimal hello integer vec panic/panic concurrency/simple pthread-threadname libc-getentropy libc-getrandom libc-misc libc-fs atomic env align num_cpus
    MIRI_TEST_TARGET=i686-unknown-freebsd run_tests_minimal hello integer vec panic/panic concurrency/simple pthread-threadname libc-getentropy libc-getrandom libc-misc libc-fs atomic env align num_cpus
    MIRI_TEST_TARGET=aarch64-linux-android run_tests_minimal hello integer vec panic/panic
    MIRI_TEST_TARGET=wasm32-wasi run_tests_minimal no_std integer strings wasm
    MIRI_TEST_TARGET=wasm32-unknown-unknown run_tests_minimal no_std integer strings wasm
    MIRI_TEST_TARGET=thumbv7em-none-eabihf run_tests_minimal no_std
    # Custom target JSON file
    MIRI_TEST_TARGET=tests/avr.json MIRI_NO_STD=1 run_tests_minimal no_std
    ;;
  aarch64-apple-darwin)
    # Host (tier 2)
    GC_STRESS=1 MIR_OPT=1 MANY_SEEDS=64 TEST_BENCH=1 CARGO_MIRI_ENV=1 run_tests
    # Extra tier 1
    MANY_SEEDS=64 MIRI_TEST_TARGET=i686-pc-windows-gnu run_tests
    MANY_SEEDS=64 MIRI_TEST_TARGET=x86_64-pc-windows-msvc CARGO_MIRI_ENV=1 run_tests
    # Extra tier 2
    MIRI_TEST_TARGET=s390x-unknown-linux-gnu run_tests # big-endian architecture
    ;;
  i686-pc-windows-msvc)
    # Host
    # Only smoke-test `many-seeds`; 64 runs of just the scoped-thread-leak test take 15min here!
    # See <https://github.com/rust-lang/miri/issues/3509>.
    GC_STRESS=1 MIR_OPT=1 MANY_SEEDS=1 TEST_BENCH=1 run_tests
    # Extra tier 1
    # We really want to ensure a Linux target works on a Windows host,
    # and a 64bit target works on a 32bit host.
    MIRI_TEST_TARGET=x86_64-unknown-linux-gnu run_tests
    ;;
  *)
    echo "FATAL: unknown host target: $HOST_TARGET"
    exit 1
    ;;
esac
