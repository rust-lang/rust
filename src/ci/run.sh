#!/bin/sh
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

if [ "$LOCAL_USER_ID" != "" ]; then
  useradd --shell /bin/bash -u $LOCAL_USER_ID -o -c "" -m user
  export HOME=/home/user
  unset LOCAL_USER_ID
  exec su --preserve-environment -c "env PATH=$PATH \"$0\"" user
fi

RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-sccache"
RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-quiet-tests"
RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --disable-manage-submodules"

# If we're deploying artifacts then we set the release channel, otherwise if
# we're not deploying then we want to be sure to enable all assertions becauase
# we'll be running tests
#
# FIXME: need a scheme for changing this `nightly` value to `beta` and `stable`
#        either automatically or manually.
if [ "$DEPLOY" != "" ]; then
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --release-channel=nightly"
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-llvm-static-stdcpp"

  if [ "$NO_LLVM_ASSERTIONS" = "1" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --disable-llvm-assertions"
  fi
else
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-debug-assertions"

  # In general we always want to run tests with LLVM assertions enabled, but not
  # all platforms currently support that, so we have an option to disable.
  if [ "$NO_LLVM_ASSERTIONS" = "" ]; then
    RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-llvm-assertions"
  fi
fi

# We want to enable usage of the `src/vendor` dir as much as possible, but not
# all test suites have all their deps in there (just the main bootstrap) so we
# have the ability to disable this flag
if [ "$NO_VENDOR" = "" ]; then
  RUST_CONFIGURE_ARGS="$RUST_CONFIGURE_ARGS --enable-vendor"
fi

set -ex

$SRC/configure $RUST_CONFIGURE_ARGS

if [ "$TRAVIS_OS_NAME" = "osx" ]; then
    ncpus=$(sysctl -n hw.ncpu)
else
    ncpus=$(grep processor /proc/cpuinfo | wc -l)
fi

if [ ! -z "$SCRIPT" ]; then
  sh -x -c "$SCRIPT"
else
  make -j $ncpus tidy
  make -j $ncpus
  make $RUST_CHECK_TARGET -j $ncpus
fi
