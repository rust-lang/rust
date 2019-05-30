#!/usr/bin/env bash

set -e

if [ -n "$CI_JOB_NAME" ]; then
  echo "[CI_JOB_NAME=$CI_JOB_NAME]"
fi

if [ "$NO_CHANGE_USER" = "" ]; then
  if [ "$LOCAL_USER_ID" != "" ]; then
    useradd --shell /bin/bash -u $LOCAL_USER_ID -o -c "" -m user
    export HOME=/home/user
    unset LOCAL_USER_ID
    exec su --preserve-environment -c "env PATH=$PATH \"$0\"" user
  fi
fi

# only enable core dump on Linux
if [ -f /proc/sys/kernel/core_pattern ]; then
  ulimit -c unlimited
fi

ci_dir=`cd $(dirname $0) && pwd`
source "$ci_dir/shared.sh"

branch_name=$(getCIBranch)

if [ ! isCI ] || [ "$branch_name" = "auto" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --set build.print-step-timings --enable-verbose-tests"
fi

RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-sccache"
RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --disable-manage-submodules"
RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-locked-deps"
RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-cargo-native-static"
RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --set rust.codegen-units-std=1"

if [ "$DIST_SRC" = "" ]; then
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --disable-dist-src"
fi

# If we're deploying artifacts then we set the release channel, otherwise if
# we're not deploying then we want to be sure to enable all assertions because
# we'll be running tests
#
# FIXME: need a scheme for changing this `nightly` value to `beta` and `stable`
#        either automatically or manually.
export RUST_RELEASE_CHANNEL=nightly
if [ "$DEPLOY$DEPLOY_ALT" = "1" ]; then
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --release-channel=$RUST_RELEASE_CHANNEL"
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-llvm-static-stdcpp"
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --set rust.remap-debuginfo"
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --set rust.debuginfo-level-std=1"

  if [ "$NO_LLVM_ASSERTIONS" = "1" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --disable-llvm-assertions"
  elif [ "$DEPLOY_ALT" != "" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-llvm-assertions"
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --set rust.verify-llvm-ir"
  fi
else
  # We almost always want debug assertions enabled, but sometimes this takes too
  # long for too little benefit, so we just turn them off.
  if [ "$NO_DEBUG_ASSERTIONS" = "" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-debug-assertions"
  fi

  # In general we always want to run tests with LLVM assertions enabled, but not
  # all platforms currently support that, so we have an option to disable.
  if [ "$NO_LLVM_ASSERTIONS" = "" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-llvm-assertions"
  fi

  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --set rust.verify-llvm-ir"
fi

if [ "$RUST_RELEASE_CHANNEL" = "nightly" ] || [ "$DIST_REQUIRE_ALL_TOOLS" = "" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-missing-tools"
fi

# We've had problems in the past of shell scripts leaking fds into the sccache
# server (#48192) which causes Cargo to erroneously think that a build script
# hasn't finished yet. Try to solve that problem by starting a very long-lived
# sccache server at the start of the build, but no need to worry if this fails.
SCCACHE_IDLE_TIMEOUT=10800 sccache --start-server || true

if [ "$RUN_CHECK_WITH_PARALLEL_QUERIES" != "" ]; then
  $SRC/configure --enable-parallel-compiler
  CARGO_INCREMENTAL=0 python2.7 ../x.py check
  rm -f config.toml
  rm -rf build
fi

travis_fold start configure
travis_time_start
$SRC/configure $RUST_CONFIGURE_ARGS
travis_fold end configure
travis_time_finish

travis_fold start make-prepare
travis_time_start
retry make prepare
travis_fold end make-prepare
travis_time_finish

travis_fold start check-bootstrap
travis_time_start
make check-bootstrap
travis_fold end check-bootstrap
travis_time_finish

# Display the CPU and memory information. This helps us know why the CI timing
# is fluctuating.
travis_fold start log-system-info
if isOSX; then
    system_profiler SPHardwareDataType || true
    sysctl hw || true
    ncpus=$(sysctl -n hw.ncpu)
else
    cat /proc/cpuinfo || true
    cat /proc/meminfo || true
    ncpus=$(grep processor /proc/cpuinfo | wc -l)
fi
travis_fold end log-system-info

if [ ! -z "$SCRIPT" ]; then
  sh -x -c "$SCRIPT"
else
  do_make() {
    travis_fold start "make-$1"
    travis_time_start
    echo "make -j $ncpus $1"
    make -j $ncpus $1
    local retval=$?
    travis_fold end "make-$1"
    travis_time_finish
    return $retval
  }

  do_make "$RUST_CHECK_TARGET"
fi

sccache --show-stats || true
