#!/usr/bin/env bash

GIT_REPO="https://github.com/rust-lang/gcc"

set -ex

cd $1

# This is the forked git commit hash used by the GCC backend.
GIT_COMMIT="$(head -n 1 libgccjit.version)"

source shared.sh

# Setting up folders for GCC
curl -L "$GIT_REPO/archive/$GIT_COMMIT.tar.gz" |
    tar -xz --transform "s/gcc-$GIT_COMMIT/gcc-src/"

mkdir gcc-build gcc-install
pushd gcc-build

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

popd
rm -rf gcc-src gcc-build
ln -s /scripts/gcc-install/lib/libgccjit.so /usr/lib/x86_64-linux-gnu/libgccjit.so
ln -s /scripts/gcc-install/lib/libgccjit.so /usr/lib/x86_64-linux-gnu/libgccjit.so.0
