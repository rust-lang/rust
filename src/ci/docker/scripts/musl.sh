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

set -ex

TAG=$1
shift

MUSL=musl-1.1.17
curl https://www.musl-libc.org/releases/$MUSL.tar.gz | tar xzf -
cd $MUSL
./configure --disable-shared --prefix=/musl-$TAG $@
make -j10
make install

cd ..
rm -rf $MUSL

# To build MUSL we're going to need a libunwind lying around, so acquire that
# here and build it.
curl -L https://github.com/llvm-mirror/llvm/archive/release_37.tar.gz | tar xzf -
curl -L https://github.com/llvm-mirror/libunwind/archive/release_37.tar.gz | tar xzf -

mkdir libunwind-build
cd libunwind-build
cmake ../libunwind-release_37 -DLLVM_PATH=/build/llvm-release_37 \
          -DLIBUNWIND_ENABLE_SHARED=0 \
          -DCMAKE_C_COMPILER=$CC \
          -DCMAKE_CXX_COMPILER=$CXX \
          -DCMAKE_C_FLAGS="$CFLAGS" \
          -DCMAKE_CXX_FLAGS="$CXXFLAGS"
make -j10
cp lib/libunwind.a /musl-$TAG/lib
