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

if [ "$NO_LLVM_ASSERTIONS" = "" ]; then
  ENABLE_LLVM_ASSERTIONS=--enable-llvm-assertions
fi

if [ "$NO_VENDOR" = "" ]; then
  ENABLE_VENDOR=--enable-vendor
fi

if [ "$NO_CCACHE" = "" ]; then
  ENABLE_CCACHE=--enable-ccache
fi

set -ex

$SRC/configure \
  --disable-manage-submodules \
  --enable-debug-assertions \
  --enable-quiet-tests \
  $ENABLE_CCACHE \
  $ENABLE_VENDOR \
  $ENABLE_LLVM_ASSERTIONS \
  $RUST_CONFIGURE_ARGS

if [ "$TRAVIS_OS_NAME" = "osx" ]; then
    ncpus=$(sysctl -n hw.ncpu)
else
    ncpus=$(nproc)
fi

make -j $ncpus tidy
make -j $ncpus
if [ ! -z "$XPY_CHECK" ]; then
  exec python2.7 $SRC/x.py $XPY_CHECK
else
  exec make $RUST_CHECK_TARGET -j $ncpus
fi
