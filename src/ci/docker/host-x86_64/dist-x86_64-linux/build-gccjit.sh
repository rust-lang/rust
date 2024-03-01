#!/bin/sh

set -ex

cd $1

# Setting up folders for GCC
git clone https://github.com/antoyo/gcc gcc-src
cd gcc-src
git checkout $(head -1 /scripts/libgccjit.version)

mkdir ../gcc-build ../gcc-install
cd ../gcc-build

# Building GCC.
../gcc-src/configure \
    --enable-host-shared \
    --enable-languages=jit \
    --enable-checking=release \
    --disable-bootstrap \
    --disable-multilib \
    --prefix=$(pwd)/../gcc-install
make
make install

rm -rf ../gcc-src
