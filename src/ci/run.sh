#!/usr/bin/env bash
# Copyright 2016 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -e

if [ "$NO_CHANGE_USER" = "" ]; then
  if [ "$LOCAL_USER_ID" != "" ]; then
    useradd --shell /bin/bash -u $LOCAL_USER_ID -o -c "" -m user
    export HOME=/home/user
    unset LOCAL_USER_ID
    exec su --preserve-environment -c "env PATH=$PATH \"$0\"" user
  fi
fi

ci_dir=`cd $(dirname $0) && pwd`
source "$ci_dir/shared.sh"

if [ "$TRAVIS" == "true" ] && [ "$TRAVIS_BRANCH" != "auto" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-quiet-tests"
fi

RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-sccache"
RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --disable-manage-submodules"
RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-locked-deps"
RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-cargo-openssl-static"

if [ "$DIST_SRC" = "" ]; then
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --disable-dist-src"
fi

# If we're deploying artifacts then we set the release channel, otherwise if
# we're not deploying then we want to be sure to enable all assertions because
# we'll be running tests
#
# FIXME: need a scheme for changing this `nightly` value to `beta` and `stable`
#        either automatically or manually.
export RUST_RELEASE_CHANNEL=stable
if [ "$DEPLOY$DEPLOY_ALT" != "" ]; then
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --release-channel=$RUST_RELEASE_CHANNEL"
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-llvm-static-stdcpp"

  if [ "$NO_LLVM_ASSERTIONS" = "1" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --disable-llvm-assertions"
  elif [ "$DEPLOY_ALT" != "" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-llvm-assertions"
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

if [ "$TRAVIS_OS_NAME" = "osx" ]; then
    ncpus=$(sysctl -n hw.ncpu)
else
    ncpus=$(grep processor /proc/cpuinfo | wc -l)
fi

if [ ! -z "$SCRIPT" ]; then
  sh -x -c "$SCRIPT"
else
  do_make() {
    travis_fold start "make-$1"
    travis_time_start
    echo "make -j $ncpus $1"
    make -j $ncpus "$1"
    local retval=$?
    travis_fold end "make-$1"
    travis_time_finish
    return $retval
  }

  do_make tidy
  do_make all
  do_make "$RUST_CHECK_TARGET"
fi
