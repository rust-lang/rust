#!/usr/bin/env bash

set -ex

source shared.sh

# Currently these commits are all tip-of-tree as of 2018-12-16, used to pick up
# a fix for rust-lang/rust#56849
LLVM=032b00a5404865765cda7db3039f39d54964d8b0
LLD=3e4aa4e8671523321af51449e0569f455ef3ad43
CLANG=a6b9739069763243020f4ea6fe586bc135fde1f9

mkdir clang
cd clang

curl -L https://github.com/llvm-mirror/llvm/archive/$LLVM.tar.gz | \
  tar xzf - --strip-components=1

mkdir -p tools/clang
curl -L https://github.com/llvm-mirror/clang/archive/$CLANG.tar.gz | \
  tar xzf - --strip-components=1 -C tools/clang

mkdir -p tools/lld
curl -L https://github.com/llvm-mirror/lld/archive/$LLD.tar.gz | \
  tar zxf - --strip-components=1 -C tools/lld

mkdir clang-build
cd clang-build

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
    cmake .. \
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
