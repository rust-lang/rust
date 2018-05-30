#!/usr/bin/env bash
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

source shared.sh

LLVM=6.0.0

mkdir clang
cd clang

curl https://releases.llvm.org/$LLVM/llvm-$LLVM.src.tar.xz | \
  xz -d | \
  tar xf -

cd llvm-$LLVM.src

mkdir -p tools/clang

curl https://releases.llvm.org/$LLVM/cfe-$LLVM.src.tar.xz | \
  xz -d | \
  tar xf - -C tools/clang --strip-components=1

mkdir -p tools/lld

curl https://releases.llvm.org/$LLVM/lld-$LLVM.src.tar.xz | \
  xz -d | \
  tar xf - -C tools/lld --strip-components=1

mkdir ../clang-build
cd ../clang-build

# For whatever reason the default set of include paths for clang is different
# than that of gcc. As a result we need to manually include our sysroot's
# include path, /rustroot/include, to clang's default include path.
#
# Alsow there's this weird oddity with gcc where there's an 'include-fixed'
# directory that it generates. It turns out [1] that Centos 5's headers are so
# old that they're incompatible with modern C semantics. While gcc automatically
# fixes that clang doesn't account for this. Tell clang to manually include the
# fixed headers so we can successfully compile code later on.
#
# [1]: https://sourceware.org/ml/crossgcc/2008-11/msg00028.html
INC="/rustroot/include"
INC="$INC:/rustroot/lib/gcc/x86_64-unknown-linux-gnu/4.8.5/include-fixed"
INC="$INC:/usr/include"

hide_output \
    cmake ../llvm-$LLVM.src \
      -DCMAKE_C_COMPILER=/rustroot/bin/gcc \
      -DCMAKE_CXX_COMPILER=/rustroot/bin/g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/rustroot \
      -DLLVM_TARGETS_TO_BUILD=X86 \
      -DC_INCLUDE_DIRS="$INC"

hide_output make -j10
hide_output make install

cd ../..
rm -rf clang
