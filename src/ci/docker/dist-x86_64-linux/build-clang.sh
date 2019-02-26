#!/usr/bin/env bash

set -ex

source shared.sh

LLVM=llvmorg-8.0.0-rc2

mkdir llvm-project
cd llvm-project

curl -L https://github.com/llvm/llvm-project/archive/$LLVM.tar.gz | \
  tar xzf - --strip-components=1

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
INC="$INC:/rustroot/lib/gcc/x86_64-unknown-linux-gnu/5.5.0/include-fixed"
INC="$INC:/usr/include"

hide_output \
    cmake ../llvm \
      -DCMAKE_C_COMPILER=/rustroot/bin/gcc \
      -DCMAKE_CXX_COMPILER=/rustroot/bin/g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/rustroot \
      -DLLVM_TARGETS_TO_BUILD=X86 \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      -DC_INCLUDE_DIRS="$INC"

hide_output make -j10
hide_output make install

cd ../..
rm -rf llvm-project
