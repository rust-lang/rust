#!/bin/sh
#
# ignore-tidy-linelength

set -ex

# Originally from https://github.com/llvm/llvm-project/releases/download/llvmorg-10.0.0/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
curl https://ci-mirrors.rust-lang.org/rustc/clang%2Bllvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz | \
  tar xJf -
export PATH=`pwd`/clang+llvm-10.0.0-x86_64-linux-gnu-ubuntu-18.04/bin:$PATH

git clone https://github.com/CraneStation/wasi-libc

cd wasi-libc
git reset --hard 9efc2f428358564fe64c374d762d0bfce1d92507
make -j$(nproc) INSTALL_DIR=/wasm32-wasi install

cd ..
rm -rf wasi-libc
rm -rf clang+llvm*
