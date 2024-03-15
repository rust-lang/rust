#!/usr/bin/env bash

set -ex

cd $1

source shared.sh

# Setting up folders for GCC
git clone https://github.com/antoyo/gcc gcc-src
cd gcc-src
# This commit hash needs to be updated to use a more recent gcc fork version.
git checkout 78dc50f0e50e6cd1433149520bd512a4e0eaa1bc

mkdir ../gcc-build ../gcc-install
cd ../gcc-build

# Building GCC.
hide_output \
  ../gcc-src/configure \
    --enable-host-shared \
    --enable-languages=jit \
    --enable-checking=release \
    --disable-bootstrap \
    --disable-multilib \
    --prefix=$(pwd)/../gcc-install \

hide_output make -j$(nproc)
hide_output make install

rm -rf ../gcc-src
ln -s /scripts/gcc-install/lib/libgccjit.so /usr/lib/x86_64-linux-gnu/libgccjit.so
ln -s /scripts/gcc-install/lib/libgccjit.so /usr/lib/x86_64-linux-gnu/libgccjit.so.0
