#!/bin/bash
set -euo pipefail

# Determine configuration
export CARGO_EXTRA_FLAGS="--all-features"
export RUSTC_EXTRA_FLAGS="-D warnings"

# Prepare
echo "Build and install miri"
./miri build --all-targets --locked
./miri install # implicitly locked
echo

# Test
function run_tests {
  if [ -n "${FOREIGN_TARGET+exists}" ]; then
    echo "Testing foreign architecture $FOREIGN_TARGET"
  else
    echo "Testing host architecture"
  fi

  ./miri test --locked
  if ! [ -n "${FOREIGN_TARGET+exists}" ]; then
    # Only for host architecture: tests with MIR optimizations
    MIRI_TEST_FLAGS="-Z mir-opt-level=3" ./miri test
  fi
  # "miri test" has built the sysroot for us, now this should pass without
  # any interactive questions.
  test-cargo-miri/run-test.py

  echo
}

# host
run_tests
# cross-test 32bit Linux from everywhere
MIRI_TEST_TARGET=i686-unknown-linux-gnu run_tests
if [ "$TRAVIS_OS_NAME" == linux ]; then
  # cross-test 64bit macOS from Linux
  FOREIGN_TARGET=x86_64-apple-darwin run_tests
fi
