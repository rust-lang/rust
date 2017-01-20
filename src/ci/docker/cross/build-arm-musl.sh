#!/bin/bash
# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -ex

MUSL=1.1.16

hide_output() {
  set +x
  on_err="
echo ERROR: An error was encountered with the build.
cat /tmp/build.log
exit 1
"
  trap "$on_err" ERR
  bash -c "while true; do sleep 30; echo \$(date) - building ...; done" &
  PING_LOOP_PID=$!
  $@ &> /tmp/build.log
  trap - ERR
  kill $PING_LOOP_PID
  rm /tmp/build.log
  set -x
}

curl -O https://www.musl-libc.org/releases/musl-$MUSL.tar.gz
tar xf musl-$MUSL.tar.gz
cd musl-$MUSL
CC=arm-linux-gnueabi-gcc \
CFLAGS="-march=armv6 -marm" \
    hide_output ./configure \
        --prefix=/usr/local/arm-linux-musleabi \
        --enable-wrapper=gcc
hide_output make -j$(nproc)
hide_output make install
cd ..
rm -rf musl-$MUSL

tar xf musl-$MUSL.tar.gz
cd musl-$MUSL
CC=arm-linux-gnueabihf-gcc \
CFLAGS="-march=armv6 -marm" \
    hide_output ./configure \
        --prefix=/usr/local/arm-linux-musleabihf \
        --enable-wrapper=gcc
hide_output make -j$(nproc)
hide_output make install
cd ..
rm -rf musl-$MUSL

tar xf musl-$MUSL.tar.gz
cd musl-$MUSL
CC=arm-linux-gnueabihf-gcc \
CFLAGS="-march=armv7-a" \
    hide_output ./configure \
        --prefix=/usr/local/armv7-linux-musleabihf \
        --enable-wrapper=gcc
hide_output make -j$(nproc)
hide_output make install
cd ..
rm -rf musl-$MUSL*

ln -nsf ../arm-linux-musleabi/bin/musl-gcc /usr/local/bin/arm-linux-musleabi-gcc
ln -nsf ../arm-linux-musleabihf/bin/musl-gcc /usr/local/bin/arm-linux-musleabihf-gcc
ln -nsf ../armv7-linux-musleabihf/bin/musl-gcc /usr/local/bin/armv7-linux-musleabihf-gcc

curl -L https://github.com/llvm-mirror/llvm/archive/release_39.tar.gz | tar xzf -
curl -L https://github.com/llvm-mirror/libunwind/archive/release_39.tar.gz | tar xzf -

mkdir libunwind-build
cd libunwind-build
cmake ../libunwind-release_39 \
          -DLLVM_PATH=/tmp/llvm-release_39 \
          -DLIBUNWIND_ENABLE_SHARED=0 \
          -DCMAKE_C_COMPILER=arm-linux-gnueabi-gcc \
          -DCMAKE_CXX_COMPILER=arm-linux-gnueabi-g++ \
          -DCMAKE_C_FLAGS="-march=armv6 -marm" \
          -DCMAKE_CXX_FLAGS="-march=armv6 -marm"
make -j$(nproc)
cp lib/libunwind.a /usr/local/arm-linux-musleabi/lib
cd ..
rm -rf libunwind-build

mkdir libunwind-build
cd libunwind-build
cmake ../libunwind-release_39 \
          -DLLVM_PATH=/tmp/llvm-release_39 \
          -DLIBUNWIND_ENABLE_SHARED=0 \
          -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
          -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
          -DCMAKE_C_FLAGS="-march=armv6 -marm" \
          -DCMAKE_CXX_FLAGS="-march=armv6 -marm"
make -j$(nproc)
cp lib/libunwind.a /usr/local/arm-linux-musleabihf/lib
cd ..
rm -rf libunwind-build

mkdir libunwind-build
cd libunwind-build
cmake ../libunwind-release_39 \
          -DLLVM_PATH=/tmp/llvm-release_39 \
          -DLIBUNWIND_ENABLE_SHARED=0 \
          -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
          -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
          -DCMAKE_C_FLAGS="-march=armv7-a" \
          -DCMAKE_CXX_FLAGS="-march=armv7-a"
make -j$(nproc)
cp lib/libunwind.a /usr/local/armv7-linux-musleabihf/lib
cd ..
rm -rf libunwind-build

rm -rf libunwind-release_39
rm -rf llvm-release_39
