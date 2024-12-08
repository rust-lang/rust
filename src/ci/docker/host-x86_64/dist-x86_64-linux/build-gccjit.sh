#!/usr/bin/env bash

GIT_REPO="https://github.com/rust-lang/gcc"

# This commit hash needs to be updated to use a more recent gcc fork version.
GIT_COMMIT="e744a9459d33864067214741daf5c5bc2a7b88c6"

set -ex

cd $1

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
