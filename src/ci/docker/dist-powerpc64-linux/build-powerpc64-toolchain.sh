#!/usr/bin/env bash
set -ex

source shared.sh

BINUTILS=2.32
TARGET=powerpc64-unknown-linux-gnu
PREFIX=/x-tools/$TARGET
SYSROOT=$PREFIX/$TARGET/sysroot

mkdir build
cd build
cp ../powerpc64-linux-gnu.config .config
hide_output ct-ng build
cd ..
rm -rf build

chmod -R u+w $PREFIX

# Next, download and build newer binutils.
mkdir binutils-$TARGET
pushd binutils-$TARGET
curl https://ftp.gnu.org/gnu/binutils/binutils-$BINUTILS.tar.bz2 | tar xjf -
mkdir binutils-build
cd binutils-build
hide_output ../binutils-$BINUTILS/configure --target=$TARGET \
  --prefix=$PREFIX --with-sysroot=$SYSROOT
hide_output make -j10
hide_output make install
popd
rm -rf binutils-$TARGET
