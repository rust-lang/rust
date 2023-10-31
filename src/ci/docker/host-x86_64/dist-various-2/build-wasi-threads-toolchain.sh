#!/bin/sh

set -ex

# Originally from https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz
curl https://ci-mirrors.rust-lang.org/rustc/2023-05-17-clang%2Bllvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04.tar.xz | \
  tar xJf -
bin="$PWD/clang+llvm-16.0.4-x86_64-linux-gnu-ubuntu-22.04/bin"

git clone https://github.com/WebAssembly/wasi-libc

cd wasi-libc
git reset --hard ec4566beae84e54952637f0bf61bee4b4cacc087
make -j$(nproc) \
    CC="$bin/clang" \
    NM="$bin/llvm-nm" \
    AR="$bin/llvm-ar" \
    THREAD_MODEL=posix \
    INSTALL_DIR=/wasm32-wasi-preview1-threads \
    install

cd ..
rm -rf wasi-libc
rm -rf clang+llvm*
