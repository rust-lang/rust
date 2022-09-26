#!/bin/bash
set -euo pipefail
set -x

# Determine configuration for installed build
echo "Installing release version of Miri"
export RUSTFLAGS="-D warnings"
export CARGO_INCREMENTAL=0
./miri install # implicitly locked

# Prepare debug build for direct `./miri` invocations
echo "Building debug version of Miri"
export CARGO_EXTRA_FLAGS="--locked"
./miri check --no-default-features # make sure this can be built
./miri check --all-features # and this, too
./miri build --all-targets # the build that all the `./miri test` below will use
echo

# Test
function run_tests {
  if [ -n "${MIRI_TEST_TARGET+exists}" ]; then
    echo "Testing foreign architecture $MIRI_TEST_TARGET"
  else
    echo "Testing host architecture"
  fi

  ## ui test suite
  ./miri test
  if [ -z "${MIRI_TEST_TARGET+exists}" ]; then
    # Only for host architecture: tests with optimizations (`-O` is what cargo passes, but crank MIR
    # optimizations up all the way).
    # Optimizations change diagnostics (mostly backtraces), so we don't check them
    #FIXME(#2155): we want to only run the pass and panic tests here, not the fail tests.
    MIRIFLAGS="${MIRIFLAGS:-} -O -Zmir-opt-level=4" MIRI_SKIP_UI_CHECKS=1 ./miri test -- tests/{pass,panic}
  fi

  ## test-cargo-miri
  # On Windows, there is always "python", not "python3" or "python2".
  if command -v python3 > /dev/null; then
    PYTHON=python3
  else
    PYTHON=python
  fi
  # Some environment setup that attempts to confuse the heck out of cargo-miri.
  if [ "$HOST_TARGET" = x86_64-unknown-linux-gnu ]; then
    # These act up on Windows (`which miri` produces a filename that does not exist?!?),
    # so let's do this only on Linux. Also makes sure things work without these set.
    export RUSTC=$(which rustc)
    export MIRI=$(which miri)
  fi
  mkdir -p .cargo
  echo 'build.rustc-wrapper = "thisdoesnotexist"' > .cargo/config.toml
  # Run the actual test
  ${PYTHON} test-cargo-miri/run-test.py
  echo
  # Clean up
  unset RUSTC MIRI
  rm -rf .cargo

  # Ensure that our benchmarks all work, but only on Linux hosts.
  if [ -z "${MIRI_TEST_TARGET+exists}" ] && [ "$HOST_TARGET" = x86_64-unknown-linux-gnu ] ; then
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

  ./miri test -- "$@"
}

# host
run_tests

case $HOST_TARGET in
  x86_64-unknown-linux-gnu)
    MIRI_TEST_TARGET=i686-unknown-linux-gnu run_tests
    MIRI_TEST_TARGET=aarch64-apple-darwin run_tests
    MIRI_TEST_TARGET=i686-pc-windows-msvc run_tests
    MIRI_TEST_TARGET=x86_64-unknown-freebsd run_tests_minimal hello integer vec panic/panic concurrency/simple atomic data_race env/var
    MIRI_TEST_TARGET=aarch64-linux-android run_tests_minimal hello integer vec panic/panic
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
