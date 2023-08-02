#!/bin/sh

set -ex

# Originally from https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
curl https://ci-mirrors.rust-lang.org/rustc/2023-05-17-clang%2Bllvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz | \
  tar xJf -
bin="$PWD/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04/bin"

git clone https://github.com/WebAssembly/wasi-libc

cd wasi-libc
git reset --hard 7018e24d8fe248596819d2e884761676f3542a04
make -j$(nproc) \
    CC="$bin/clang" \
    NM="$bin/llvm-nm" \
    AR="$bin/llvm-ar" \
    INSTALL_DIR=/wasm32-wasi \
    install

cd ..
rm -rf wasi-libc
rm -rf clang+llvm*
