#!/bin/sh
#
# ignore-tidy-linelength

set -ex

# Originally from https://releases.llvm.org/8.0.0/clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-14.04.tar.xz
curl https://rust-lang-ci2.s3.amazonaws.com/rust-ci-mirror/clang%2Bllvm-8.0.0-x86_64-linux-gnu-ubuntu-14.04.tar.xz | \
  tar xJf -
export PATH=`pwd`/clang+llvm-8.0.0-x86_64-linux-gnu-ubuntu-14.04/bin:$PATH

git clone https://github.com/CraneStation/wasi-sysroot

cd wasi-sysroot
git reset --hard e5f14be38362f1ab83302895a6e74b2ffd0e2302
make -j$(nproc) INSTALL_DIR=/wasm32-wasi install

cd ..
rm -rf reference-sysroot-wasi
rm -rf clang+llvm*
