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

# Determine configuration for installed build
echo "Installing release version of Miri"
export RUSTFLAGS="-D warnings"
export CARGO_INCREMENTAL=0
export CARGO_EXTRA_FLAGS="--locked"
./miri install

# Prepare debug build for direct `./miri` invocations
echo "Building debug version of Miri"
./miri check --no-default-features # make sure this can be built
./miri check --all-features # and this, too
./miri build --all-targets # the build that all the `./miri test` below will use

endgroup

# Test
function run_tests {
  if [ -n "${MIRI_TEST_TARGET+exists}" ]; then
    begingroup "Testing foreign architecture $MIRI_TEST_TARGET"
  else
    begingroup "Testing host architecture"
  fi

  ## ui test suite
  ./miri test
  if [ -z "${MIRI_TEST_TARGET+exists}" ]; then
    # Host-only tests: running these on all targets is unlikely to catch more problems and would
    # cost a lot of CI time.

    # Tests with optimizations (`-O` is what cargo passes, but crank MIR optimizations up all the
    # way, too).
    # Optimizations change diagnostics (mostly backtraces), so we don't check
    # them. Also error locations change so we don't run the failing tests.
    # We explicitly enable debug-assertions here, they are disabled by -O but we have tests
    # which exist to check that we panic on debug assertion failures.
    MIRIFLAGS="${MIRIFLAGS:-} -O -Zmir-opt-level=4 -Cdebug-assertions=yes" MIRI_SKIP_UI_CHECKS=1 ./miri test -- tests/{pass,panic}

    # Also run some many-seeds tests. 64 seeds means this takes around a minute per test.
    for FILE in tests/many-seeds/*.rs; do
      MIRI_SEEDS=64 CARGO_EXTRA_FLAGS="$CARGO_EXTRA_FLAGS -q" ./miri many-seeds ./miri run "$FILE"
    done

    # Check that the benchmarks build and run, but without actually benchmarking.
    HYPERFINE="bash -c" ./miri bench
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
    export RUSTC=$(which rustc) # Produces a warning unless we also set MIRI
    export MIRI=$(rustc +miri --print sysroot)/bin/miri
  fi
  mkdir -p .cargo
  echo 'build.rustc-wrapper = "thisdoesnotexist"' > .cargo/config.toml
  # Run the actual test
  ${PYTHON} test-cargo-miri/run-test.py
  # Clean up
  unset RUSTC MIRI
  rm -rf .cargo

  endgroup
}

function run_tests_minimal {
  if [ -n "${MIRI_TEST_TARGET+exists}" ]; then
    begingroup "Testing MINIMAL foreign architecture $MIRI_TEST_TARGET: only testing $@"
  else
    begingroup "Testing MINIMAL host architecture: only testing $@"
  fi

  ./miri test -- "$@"

  # Ensure that a small smoke test of cargo-miri works.
  cargo miri run --manifest-path test-cargo-miri/no-std-smoke/Cargo.toml --target ${MIRI_TEST_TARGET-$HOST_TARGET}

  endgroup
}

# host
run_tests

case $HOST_TARGET in
  x86_64-unknown-linux-gnu)
    MIRI_TEST_TARGET=i686-unknown-linux-gnu run_tests
    MIRI_TEST_TARGET=aarch64-unknown-linux-gnu run_tests
    MIRI_TEST_TARGET=aarch64-apple-darwin run_tests
    MIRI_TEST_TARGET=i686-pc-windows-msvc run_tests
    MIRI_TEST_TARGET=x86_64-unknown-freebsd run_tests_minimal hello integer vec panic/panic concurrency/simple atomic data_race env/var
    MIRI_TEST_TARGET=aarch64-linux-android run_tests_minimal hello integer vec panic/panic
    MIRI_TEST_TARGET=wasm32-wasi run_tests_minimal no_std integer strings
    MIRI_TEST_TARGET=wasm32-unknown-unknown run_tests_minimal no_std integer strings
    MIRI_TEST_TARGET=thumbv7em-none-eabihf MIRI_NO_STD=1 run_tests_minimal no_std # no_std embedded architecture
    MIRI_TEST_TARGET=tests/avr.json MIRI_NO_STD=1 run_tests_minimal no_std # JSON target file
    ;;
  x86_64-apple-darwin)
    MIRI_TEST_TARGET=mips64-unknown-linux-gnuabi64 run_tests # big-endian architecture
    MIRI_TEST_TARGET=x86_64-pc-windows-msvc run_tests
    ;;
  i686-pc-windows-msvc)
    MIRI_TEST_TARGET=arm-unknown-linux-gnueabi run_tests
    MIRI_TEST_TARGET=x86_64-unknown-linux-gnu run_tests
    MIRI_TEST_TARGET=x86_64-pc-windows-gnu run_tests
    ;;
  *)
    echo "FATAL: unknown OS"
    exit 1
    ;;
esac
