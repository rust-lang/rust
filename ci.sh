#!/bin/bash
set -euo pipefail

# Determine configuration
export RUSTFLAGS="-D warnings"
export CARGO_INCREMENTAL=0
export CARGO_EXTRA_FLAGS="--all-features"

# Prepare
echo "Build and install miri"
./miri build --all-targets --locked
./miri install # implicitly locked
echo

# Test
function run_tests {
  if [ -n "${MIRI_TEST_TARGET+exists}" ]; then
    echo "Testing foreign architecture $MIRI_TEST_TARGET"
  else
    echo "Testing host architecture"
  fi

  ./miri test --locked
  if [ -z "${MIRI_TEST_TARGET+exists}" ]; then
    # Only for host architecture: tests with optimizations (`-O` is what cargo passes, but crank MIR
    # optimizations up all the way).
    # Optimizations change diagnostics (mostly backtraces), so we don't check them
    #FIXME(#2155): we want to only run the pass and panic tests here, not the fail tests.
    MIRIFLAGS="-O -Zmir-opt-level=4" MIRI_SKIP_UI_CHECKS=1 ./miri test --locked -- tests/{pass,panic}
  fi

  # On Windows, there is always "python", not "python3" or "python2".
  if command -v python3 > /dev/null; then
    PYTHON=python3
  else
    PYTHON=python
  fi

  # "miri test" has built the sysroot for us, now this should pass without
  # any interactive questions.
  ${PYTHON} test-cargo-miri/run-test.py
  echo

  # Ensure that our benchmarks all work, on the host at least.
  if [ -z "${MIRI_TEST_TARGET+exists}" ]; then
    for BENCH in $(ls "bench-cargo-miri"); do
      cargo miri run --manifest-path bench-cargo-miri/$BENCH/Cargo.toml
    done
  fi
}

function run_tests_minimal {
  if [ -n "${MIRI_TEST_TARGET+exists}" ]; then
    echo "Testing MINIMAL foreign architecture $MIRI_TEST_TARGET: only testing $@"
  else
    echo "Testing MINIMAL host architecture: only testing $@"
  fi

  ./miri test --locked -- "$@"
}

# host
run_tests

case $HOST_TARGET in
  x86_64-unknown-linux-gnu)
    MIRI_TEST_TARGET=i686-unknown-linux-gnu run_tests
    MIRI_TEST_TARGET=aarch64-apple-darwin run_tests
    MIRI_TEST_TARGET=i686-pc-windows-msvc run_tests
    MIRI_TEST_TARGET=x86_64-unknown-freebsd run_tests_minimal hello integer vec current_dir data_race env
    MIRI_TEST_TARGET=thumbv7em-none-eabihf MIRI_NO_STD=1 run_tests_minimal no_std # no_std embedded architecture
    ;;
  x86_64-apple-darwin)
    MIRI_TEST_TARGET=mips64-unknown-linux-gnuabi64 run_tests # big-endian architecture
    MIRI_TEST_TARGET=x86_64-pc-windows-msvc run_tests
    ;;
  i686-pc-windows-msvc)
    MIRI_TEST_TARGET=x86_64-unknown-linux-gnu run_tests
    ;;
  *)
    echo "FATAL: unknown OS"
    exit 1
    ;;
esac
