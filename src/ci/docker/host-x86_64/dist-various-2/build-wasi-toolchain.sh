#!/bin/sh
#
# ignore-tidy-linelength

set -ex

# Originally from https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
curl https://ci-mirrors.rust-lang.org/rustc/clang%2Bllvm-11.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz | \
  tar xJf -
export PATH=`pwd`/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-16.04/bin:$PATH

git clone https://github.com/WebAssembly/wasi-libc

cd wasi-libc
git reset --hard 58795582905e08fa7748846c1971b4ab911d1e16
make -j$(nproc) INSTALL_DIR=/wasm32-wasi install

cd ..
rm -rf wasi-libc
rm -rf clang+llvm*
