#!/bin/bash
set -euo pipefail

# Determine configuration
if [ "$TRAVIS_OS_NAME" == osx ]; then
  MIRI_SYSROOT_BASE=~/Library/Caches/miri.miri.miri/
  FOREIGN_TARGET=i686-apple-darwin
else
  MIRI_SYSROOT_BASE=~/.cache/miri/
  FOREIGN_TARGET=i686-unknown-linux-gnu
fi

# Prepare
echo "Build and install miri"
cargo build --release --all-features --all-targets
cargo install --all-features --force --path .
echo

echo "Get ourselves a MIR-full libstd for the host and a foreign architecture"
cargo miri setup
cargo miri setup --target "$FOREIGN_TARGET"
echo

# Test
function run_tests {
  cargo test --release --all-features
  test-cargo-miri/run-test.py
}

echo "Test host architecture"
export MIRI_SYSROOT="$MIRI_SYSROOT_BASE"/HOST
run_tests
echo

echo "Test foreign architecture ($FOREIGN_TARGET)"
export MIRI_SYSROOT="$MIRI_SYSROOT_BASE" MIRI_TEST_TARGET="$FOREIGN_TARGET"
run_tests
echo
