#!/usr/bin/env bash

set -ex

source shared.sh

LLVM=llvmorg-9.0.0

mkdir llvm-project
cd llvm-project

curl -L https://github.com/llvm/llvm-project/archive/$LLVM.tar.gz | \
  tar xzf - --strip-components=1

yum install -y patch
patch -Np1 < ../llvm-project-centos.patch

mkdir clang-build
cd clang-build

hide_output \
    cmake ../llvm \
      -DCMAKE_C_COMPILER=/rustroot/bin/gcc \
      -DCMAKE_CXX_COMPILER=/rustroot/bin/g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/rustroot \
      -DLLVM_TARGETS_TO_BUILD=X86 \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      -DGCC_INSTALL_PREFIX=/rustroot

hide_output make -j10
hide_output make install

cd ../..
rm -rf llvm-project
