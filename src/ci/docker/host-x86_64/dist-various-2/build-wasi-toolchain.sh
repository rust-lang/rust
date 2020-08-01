#!/bin/sh
#
# ignore-tidy-linelength

set -ex

# Originally from https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
curl https://ci-mirrors.rust-lang.org/rustc/clang%2Bllvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz | \
  tar xJf -
export PATH=`pwd`/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH

git clone https://github.com/WebAssembly/wasi-libc

cd wasi-libc
git reset --hard 215adc8ac9f91eb055311acc72683fd2eb1ae15a
make -j$(nproc) INSTALL_DIR=/wasm32-wasi install

cd ..
rm -rf wasi-libc
rm -rf clang+llvm*
