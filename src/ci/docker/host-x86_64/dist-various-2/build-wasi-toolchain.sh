#!/bin/sh

set -ex

# Originally from https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
curl https://ci-mirrors.rust-lang.org/rustc/2021-01-14-clang%2Bllvm-11.0.1-x86_64-linux-gnu-ubuntu-16.04.tar.xz | \
  tar xJf -
export PATH=`pwd`/clang+llvm-11.0.1-x86_64-linux-gnu-ubuntu-16.04/bin:$PATH

git clone https://github.com/WebAssembly/wasi-libc

cd wasi-libc
git reset --hard ad5133410f66b93a2381db5b542aad5e0964db96
make -j$(nproc) INSTALL_DIR=/wasm32-wasi install

cd ..
rm -rf wasi-libc
rm -rf clang+llvm*
