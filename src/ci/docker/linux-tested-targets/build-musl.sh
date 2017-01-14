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

export CFLAGS="-fPIC"
MUSL=musl-1.1.14
curl https://www.musl-libc.org/releases/$MUSL.tar.gz | tar xzf -
cd $MUSL
./configure --prefix=/musl-x86_64 --disable-shared
make -j10
make install
make clean
# for i686
CFLAGS="$CFLAGS -m32" ./configure --prefix=/musl-i686 --disable-shared --target=i686
make -j10
make install
cd ..

# To build MUSL we're going to need a libunwind lying around, so acquire that
# here and build it.
curl -L https://github.com/llvm-mirror/llvm/archive/release_37.tar.gz | tar xzf -
curl -L https://github.com/llvm-mirror/libunwind/archive/release_37.tar.gz | tar xzf -

# Whoa what's this mysterious patch we're applying to libunwind! Why are we
# swapping the values of ESP/EBP in libunwind?!
#
# Discovered in #35599 it turns out that the vanilla build of libunwind is not
# suitable for unwinding 32-bit musl. After some investigation it ended up
# looking like the register values for ESP/EBP were indeed incorrect (swapped)
# in the source. Similar commits in libunwind (r280099 and r282589) have noticed
# this for other platforms, and we just need to realize it for musl linux as
# well.
#
# More technical info can be found at #35599
cd libunwind-release_37
patch -Np1 < /build/musl-libunwind-patch.patch
cd ..

mkdir libunwind-build
cd libunwind-build
cmake ../libunwind-release_37 -DLLVM_PATH=/build/llvm-release_37 \
          -DLIBUNWIND_ENABLE_SHARED=0
make -j10
cp lib/libunwind.a /musl-x86_64/lib

# (Note: the next cmake call doesn't fully override the previous cached one, so remove the cached
# configuration manually. IOW, if don't do this or call make clean we'll end up building libunwind
# for x86_64 again)
rm -rf *
# for i686
CFLAGS="$CFLAGS -m32 -g" CXXFLAGS="$CXXFLAGS -m32 -g" cmake ../libunwind-release_37 \
          -DLLVM_PATH=/build/llvm-release_37 \
          -DLIBUNWIND_ENABLE_SHARED=0
make -j10
cp lib/libunwind.a /musl-i686/lib
