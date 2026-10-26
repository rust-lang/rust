#!/usr/bin/env bash
# Since FreeBSD stores user-installed binaries in `/usr/local/bin` we
# cannot use the `/bin/bash` shebang and instead use the the bash from PATH.

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

export CI=true # just to be sure that it is truly set

endgroup

begingroup "Building Miri"

# Global configuration
export CARGO_BUILD_WARNINGS=deny
export CARGO_INCREMENTAL=0
export CARGO_EXTRA_FLAGS="--locked"
export CARGO_UNSTABLE_BUILD_DIR_NEW_LAYOUT=true

# Determine configuration for installed build (used by test-cargo-miri and `./miri bench`).
# We use the default set of features for this.
echo "Installing release version of Miri"
time ./miri install

# Prepare debug build for direct `./miri` invocations.
# Here we enable some more features and checks.
echo "Building debug version of Miri"
export FEATURES="--features=expensive-consistency-checks,genmc"
time ./miri build $FEATURES # the build that all the `./miri test` below will use

# We don't want to error on warnings during the sysroot build, so unset this.
# (Bootstrap sets this so the tier 1 targets are checked in rustc CI.)
unset CARGO_BUILD_WARNINGS

endgroup

# Run tests. Recognizes these variables:
# - TEST_TARGET: the target to test. Empty for host target.
# - TEST_TIER: configure which and how many tests to run.
#   Tier 2: many-seeds tests run 16 times, nothing extra
#   Tier 1: many-seeds tests run 64 times, check `./miri run`
#   Tier 0: tier 1 checks plus mir-opt, test suite runs with GC stress tests,
#   check that benchmarks build, and mess with cargo-miri configuration
function run_tests {
  if [ -n "${TEST_TARGET-}" ]; then
    begingroup "Testing foreign architecture $TEST_TARGET (at tier $TEST_TIER)"
    TARGET_FLAG="--target $TEST_TARGET"
    MULTI_TARGET_FLAG=""
  else
    begingroup "Testing host architecture (at tier $TEST_TIER)"
    TARGET_FLAG=""
    MULTI_TARGET_FLAG="--multi-target"
  fi

  ## ui test suite
  if (( $TEST_TIER <= 0 )); then
    # We set this only for  the `./miri test`, not the rest
    MIRIFLAGS_EXTRA="-Zmiri-provenance-gc=1"
  fi
  time MIRIFLAGS="${MIRIFLAGS-} ${MIRIFLAGS_EXTRA-}" ./miri test $FEATURES $TARGET_FLAG
  unset MIRIFLAGS_EXTRA

  ## mir-opt tests
  if (( $TEST_TIER <= 0 )); then
    # Run tests with optimizations. `-O` is what cargo passes, but crank MIR optimizations up all
    # the way, too. Optimizations change diagnostics (mostly backtraces), so we don't check them.
    # Also error locations change so we don't run the failing tests. We explicitly enable
    # debug-assertions here, they are disabled by -O but we have tests which exist to check that we
    # panic on debug assertion failures.
    time MIRIFLAGS="${MIRIFLAGS-} -O -Zmir-opt-level=4 -Cdebug-assertions=yes" MIRI_SKIP_UI_CHECKS=1 ./miri test $FEATURES $TARGET_FLAG tests/{pass,panic}
  fi
  ## many-seeds tests
  if (( $TEST_TIER <= 1 )); then
    MANY_SEEDS=64
  else
    MANY_SEEDS=16
  fi
  time for FILE in tests/many-seeds/*.rs; do
    ./miri run $FEATURES "-Zmiri-many-seeds=0..$MANY_SEEDS" $TARGET_FLAG "$FILE"
  done
  ## `./miri run` (including `--native`, but that only works for the host)
  if (( $TEST_TIER <= 1 )); then
    time ./miri run $FEATURES $TARGET_FLAG --dep tests/pass-dep/getrandom.rs
  fi
  if [ -z "${TEST_TARGET-}" ]; then
    time ./miri run $FEATURES --native tests/pass-dep/getrandom.rs
  fi
  ## benchmarks
  if (( $TEST_TIER <= 0 )); then
    time HYPERFINE="hyperfine -w0 -r1 --show-output" ./miri bench $TARGET_FLAG --no-install
  fi

  ## test-cargo-miri
  # On Windows, there is always "python", not "python3" or "python2".
  if command -v python3 > /dev/null; then
    PYTHON=python3
  else
    PYTHON=python
  fi
  # Some environment setup that attempts to confuse the heck out of cargo-miri.
  if (( $TEST_TIER == 0 )); then
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
  time cargo miri run --manifest-path test-cargo-miri/no-std-smoke/Cargo.toml -Zjson-target-spec $TARGET_FLAG

  endgroup
}

## Main Testing Logic ##

# Key constraints:
# - all tier 1 targets covered with TEST_TIER=1, plus s390x-unknown-linux-gnu
# - every tier 1 OS & arch covered with TEST_TIER=0 ("extended tests")
case $HOST_TARGET in
  x86_64-unknown-linux-gnu)
    TEST_TIER=0 run_tests # Linux, x86_64 extended tests
    TEST_TIER=1 TEST_TARGET=x86_64-pc-windows-gnu run_tests # Windows on Linux
    ;;
  i686-unknown-linux-gnu)
    TEST_TIER=0 run_tests # i686 extended tests
    TEST_TIER=2 TEST_TARGET=aarch64-linux-android run_tests # 64bit on 32bit
    ;;
  aarch64-unknown-linux-gnu)
    TEST_TIER=1 run_tests
    TEST_TIER=2 TEST_TARGET=loongarch64-unknown-linux-gnu run_tests
    TEST_TIER=2 TEST_TARGET=riscv64a23-unknown-linux-gnu run_tests
    TEST_TIER=2 TEST_TARGET=armv7-unknown-linux-gnueabihf run_tests
    TEST_TIER=2 TEST_TARGET=i686-unknown-freebsd run_tests # 32bit on 64bit
    TEST_TIER=2 TEST_TARGET=mips-unknown-linux-gnu run_tests # a 32bit big-endian target, and also a target without 64bit atomics
    TEST_TIER=2 TEST_TARGET=x86_64-unknown-illumos run_tests
    TEST_TIER=2 TEST_TARGET=i686-pc-windows-gnu run_tests
    TEST_TARGET=wasm32-unknown-unknown run_tests_minimal no_std empty_main wasm # this target doesn't really have std
    TEST_TARGET=thumbv7em-none-eabihf  run_tests_minimal no_std
    TEST_TARGET=x86_64-unknown-netbsd run_tests_minimal hello libc/env.rs libc/misc.rs
    ;;
  aarch64-apple-darwin)
    TEST_TIER=0 run_tests # macOS, aarch64 extended tests
    TEST_TIER=2 TEST_TARGET=i686-pc-windows-msvc run_tests
    ;;
  x86_64-pc-windows-msvc)
    TEST_TIER=0 run_tests # Windows extended tests
    TEST_TIER=2 TEST_TARGET=x86_64-apple-darwin run_tests
    ;;
  aarch64-pc-windows-msvc)
    TEST_TIER=1 run_tests
    TEST_TIER=1 TEST_TARGET=s390x-unknown-linux-gnu run_tests # big-endian, Linux on Windows
    TEST_TIER=2 TEST_TARGET=x86_64-pc-solaris run_tests
    ;;
  x86_64-unknown-freebsd)
    TEST_TIER=2 run_tests
    # Don't put Linux targets here, they need address 127.0.1.1 which the VM does not seem to have.
    TEST_TIER=2 TEST_TARGET=x86_64-pc-windows-gnullvm run_tests
    TEST_TARGET=tests/x86_64-unknown-kernel.json MIRI_NO_STD=1 MIRIFLAGS="-Zunstable-options" run_tests_minimal no_std # JSON target
    ;;
  *)
    echo "FATAL: unknown host target: $HOST_TARGET"
    exit 1
    ;;
esac
