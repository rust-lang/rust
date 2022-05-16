#!/bin/sh

set -ex

# Originally from https://github.com/llvm/llvm-project/releases/download/llvmorg-14.0.0/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
curl https://ci-mirrors.rust-lang.org/rustc/2022-05-10-clang%2Bllvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz | \
  tar xJf -
bin="$PWD/clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/bin"

git clone https://github.com/WebAssembly/wasi-libc

cd wasi-libc
git reset --hard 9886d3d6200fcc3726329966860fc058707406cd
make -j$(nproc) \
    CC="$bin/clang" \
    NM="$bin/llvm-nm" \
    AR="$bin/llvm-ar" \
    INSTALL_DIR=/wasm32-wasi \
    install

cd ..
rm -rf wasi-libc
rm -rf clang+llvm*
