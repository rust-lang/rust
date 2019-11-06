#!/bin/sh
#
# ignore-tidy-linelength

set -ex

# Originally from https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-14.04.tar.xz
curl https://rust-lang-ci-mirrors.s3-us-west-1.amazonaws.com/rustc/clang%2Bllvm-9.0.0-x86_64-linux-gnu-ubuntu-14.04.tar.xz | \
  tar xJf -
export PATH=`pwd`/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-14.04/bin:$PATH

git clone https://github.com/CraneStation/wasi-libc

cd wasi-libc
git reset --hard a94d2d04e7722b323573da2bd04e909a5763d35b
make -j$(nproc) INSTALL_DIR=/wasm32-wasi install

cd ..
rm -rf wasi-libc
rm -rf clang+llvm*
