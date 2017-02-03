#!/bin/bash
# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -ex

BINUTILS=2.25.1
GCC=5.3.0
TARGET=powerpc64le-linux-gnu
SYSROOT=/usr/local/$TARGET/sysroot

# First, download the CentOS7 glibc.ppc64le and relevant header files.
# (upstream ppc64le support wasn't added until 2.19, which el7 backported.)
mkdir -p $SYSROOT
pushd $SYSROOT

centos_base=http://mirror.centos.org/altarch/7.3.1611/os/ppc64le/Packages
glibc_v=2.17-157.el7
kernel_v=3.10.0-514.el7
for package in glibc{,-devel,-headers}-$glibc_v kernel-headers-$kernel_v; do
  curl $centos_base/$package.ppc64le.rpm | \
    rpm2cpio - | cpio -idm
done

ln -sT lib64 lib
ln -sT lib64 usr/lib

popd

# Next, download and build binutils.
mkdir binutils-$TARGET
pushd binutils-$TARGET
curl https://ftp.gnu.org/gnu/binutils/binutils-$BINUTILS.tar.bz2 | tar xjf -
mkdir binutils-build
cd binutils-build
../binutils-$BINUTILS/configure --target=$TARGET --with-sysroot=$SYSROOT
make -j10
make install
popd
rm -rf binutils-$TARGET

# Finally, download and build gcc.
mkdir gcc-$TARGET
pushd gcc-$TARGET
curl https://ftp.gnu.org/gnu/gcc/gcc-$GCC/gcc-$GCC.tar.bz2 | tar xjf -
cd gcc-$GCC
./contrib/download_prerequisites

mkdir ../gcc-build
cd ../gcc-build
../gcc-$GCC/configure                            \
  --enable-languages=c,c++                       \
  --target=$TARGET                               \
  --with-cpu=power8                              \
  --with-sysroot=$SYSROOT                        \
  --disable-libcilkrts                           \
  --disable-multilib                             \
  --disable-nls                                  \
  --disable-libgomp                              \
  --disable-libquadmath                          \
  --disable-libssp                               \
  --disable-libvtv                               \
  --disable-libcilkrt                            \
  --disable-libada                               \
  --disable-libsanitizer                         \
  --disable-libquadmath-support                  \
  --disable-lto
make -j10
make install

popd
rm -rf gcc-$TARGET
