#!/bin/sh

set -ex

# Originally from https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.6/clang+llvm-15.0.6-x86_64-linux-gnu-ubuntu-18.04.tar.xz
curl https://ci-mirrors.rust-lang.org/rustc/2022-12-06-clang%2Bllvm-15.0.6-x86_64-linux-gnu-ubuntu-18.04.tar.xz | \
  tar xJf -
bin="$PWD/clang+llvm-15.0.6-x86_64-linux-gnu-ubuntu-18.04/bin"

git clone https://github.com/WebAssembly/wasi-libc

cd wasi-libc
git reset --hard 8b7148f69ae241a2749b3defe4606da8143b72e0
make -j$(nproc) \
    CC="$bin/clang" \
    NM="$bin/llvm-nm" \
    AR="$bin/llvm-ar" \
    INSTALL_DIR=/wasm32-wasi \
    install

cd ..
rm -rf wasi-libc
rm -rf clang+llvm*
