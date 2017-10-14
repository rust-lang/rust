#!/bin/bash
# Copyright 2016 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

set -ex
source shared.sh

ARCH=$1
LIB_ARCH=$2
APT_ARCH=$3
BINUTILS=2.28.1
GCC=6.4.0

# First up, build binutils
mkdir binutils
cd binutils

curl https://ftp.gnu.org/gnu/binutils/binutils-$BINUTILS.tar.xz | tar xJf -
mkdir binutils-build
cd binutils-build
hide_output ../binutils-$BINUTILS/configure --target=$ARCH-sun-solaris2.10
hide_output make -j10
hide_output make install

cd ../..
rm -rf binutils

# Next, download and install the relevant solaris packages
mkdir solaris
cd solaris

dpkg --add-architecture $APT_ARCH
apt-get update
apt-get download           \
  libc:$APT_ARCH           \
  libc-dev:$APT_ARCH       \
  libm:$APT_ARCH           \
  libm-dev:$APT_ARCH       \
  libpthread:$APT_ARCH     \
  libpthread-dev:$APT_ARCH \
  libresolv:$APT_ARCH      \
  libresolv-dev:$APT_ARCH  \
  librt:$APT_ARCH          \
  librt-dev:$APT_ARCH      \
  libsocket:$APT_ARCH      \
  libsocket-dev:$APT_ARCH  \
  system-crt:$APT_ARCH     \
  system-header:$APT_ARCH

for deb in *$APT_ARCH.deb; do
  dpkg -x $deb .
done

# Strip Solaris 11 functions that are optionally used by libbacktrace.
# This is for Solaris 10 compatibility.
$ARCH-sun-solaris2.10-strip -N dl_iterate_phdr -N strnlen lib/$LIB_ARCH/libc.so

mkdir                  /usr/local/$ARCH-sun-solaris2.10/usr
mv usr/include         /usr/local/$ARCH-sun-solaris2.10/usr/include
mv usr/lib/$LIB_ARCH/* /usr/local/$ARCH-sun-solaris2.10/lib
mv     lib/$LIB_ARCH/* /usr/local/$ARCH-sun-solaris2.10/lib

ln -s /usr/local/$ARCH-sun-solaris2.10/usr/include /usr/local/$ARCH-sun-solaris2.10/sys-include
ln -s /usr/local/$ARCH-sun-solaris2.10/usr/include /usr/local/$ARCH-sun-solaris2.10/include

cd ..
rm -rf solaris

# Finally, download and build gcc to target solaris
mkdir gcc
cd gcc

curl https://ftp.gnu.org/gnu/gcc/gcc-$GCC/gcc-$GCC.tar.xz | tar xJf -
cd gcc-$GCC

mkdir ../gcc-build
cd ../gcc-build
hide_output ../gcc-$GCC/configure \
  --enable-languages=c,c++        \
  --target=$ARCH-sun-solaris2.10  \
  --with-gnu-as                   \
  --with-gnu-ld                   \
  --disable-multilib              \
  --disable-nls                   \
  --disable-libgomp               \
  --disable-libquadmath           \
  --disable-libssp                \
  --disable-libvtv                \
  --disable-libcilkrts            \
  --disable-libada                \
  --disable-libsanitizer          \
  --disable-libquadmath-support   \
  --disable-lto                   \
  --with-sysroot=/usr/local/$ARCH-sun-solaris2.10

hide_output make -j10
hide_output make install

cd ../..
rm -rf gcc
