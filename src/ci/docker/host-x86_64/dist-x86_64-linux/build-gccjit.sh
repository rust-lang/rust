#!/usr/bin/env bash

GIT_REPO="https://github.com/antoyo/gcc"

# This commit hash needs to be updated to use a more recent gcc fork version.
GIT_COMMIT="78dc50f0e50e6cd1433149520bd512a4e0eaa1bc"

set -ex

cd $1

source shared.sh

# Setting up folders for GCC
curl -L "$GIT_REPO/archive/$GIT_COMMIT.tar.gz" |
    tar -xz --transform "s/gcc-$GIT_COMMIT/gcc-src/"

mkdir gcc-build gcc-install
cd gcc-build

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

cd ..
rm -rf gcc-src gcc-build
ln -s /scripts/gcc-install/lib/libgccjit.so /usr/lib/x86_64-linux-gnu/libgccjit.so
ln -s /scripts/gcc-install/lib/libgccjit.so /usr/lib/x86_64-linux-gnu/libgccjit.so.0
