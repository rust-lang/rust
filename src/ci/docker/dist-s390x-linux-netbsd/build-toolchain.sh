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

# ignore-tidy-linelength

set -ex

BINUTILS=2.25.1
GCC=5.3.0

# First up, build binutils
mkdir binutils
cd binutils
curl https://ftp.gnu.org/gnu/binutils/binutils-$BINUTILS.tar.bz2 | tar xjf -
mkdir binutils-build
cd binutils-build
../binutils-$BINUTILS/configure \
  --target=x86_64-unknown-netbsd
make -j10
make install
cd ../..
rm -rf binutils

# Next, download the NetBSD libc and relevant header files
mkdir netbsd
# originally from:
# https://ftp.netbsd.org/pub/NetBSD/NetBSD-7.0/amd64/binary/sets/base.tgz
curl https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-01-16-netbsd-base.tgz | \
  tar xzf - -C netbsd ./usr/include ./usr/lib ./lib
# originally from:
# https://ftp.netbsd.org/pub/NetBSD/NetBSD-7.0/amd64/binary/sets/comp.tgz
curl https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-01-16-netbsd-comp.tgz | \
  tar xzf - -C netbsd ./usr/include ./usr/lib

dst=/usr/local/x86_64-unknown-netbsd
cp -r netbsd/usr/include $dst
cp netbsd/usr/lib/crt0.o $dst/lib
cp netbsd/usr/lib/crti.o $dst/lib
cp netbsd/usr/lib/crtn.o $dst/lib
cp netbsd/usr/lib/crtbeginS.o $dst/lib
cp netbsd/usr/lib/crtendS.o $dst/lib
cp netbsd/usr/lib/crtbegin.o $dst/lib
cp netbsd/usr/lib/crtend.o $dst/lib
cp netbsd/usr/lib/gcrt0.o $dst/lib
cp netbsd/usr/lib/libc.a $dst/lib
cp netbsd/usr/lib/libc_p.a $dst/lib
cp netbsd/usr/lib/libc_pic.a $dst/lib
cp netbsd/lib/libc.so.12.193.1 $dst/lib
cp netbsd/lib/libutil.so.7.21 $dst/lib
cp netbsd/usr/lib/libm.a $dst/lib
cp netbsd/usr/lib/libm_p.a $dst/lib
cp netbsd/usr/lib/libm_pic.a $dst/lib
cp netbsd/lib/libm.so.0.11 $dst/lib
cp netbsd/usr/lib/librt.so.1.1 $dst/lib
cp netbsd/usr/lib/libpthread.a $dst/lib
cp netbsd/usr/lib/libpthread_p.a $dst/lib
cp netbsd/usr/lib/libpthread_pic.a $dst/lib
cp netbsd/usr/lib/libpthread.so.1.2 $dst/lib

ln -s libc.so.12.193.1 $dst/lib/libc.so
ln -s libc.so.12.193.1 $dst/lib/libc.so.12
ln -s libm.so.0.11 $dst/lib/libm.so
ln -s libm.so.0.11 $dst/lib/libm.so.0
ln -s libutil.so.7.21 $dst/lib/libutil.so
ln -s libutil.so.7.21 $dst/lib/libutil.so.7
ln -s libpthread.so.1.2 $dst/lib/libpthread.so
ln -s libpthread.so.1.2 $dst/lib/libpthread.so.1
ln -s librt.so.1.1 $dst/lib/librt.so

rm -rf netbsd

# Finally, download and build gcc to target NetBSD
mkdir gcc
cd gcc
curl https://ftp.gnu.org/gnu/gcc/gcc-$GCC/gcc-$GCC.tar.bz2 | tar xjf -
cd gcc-$GCC
./contrib/download_prerequisites

# Originally from
# ftp://ftp.netbsd.org/pub/pkgsrc/pkgsrc-2016Q4/pkgsrc/lang/gcc5/patches/patch-libstdc%2B%2B-v3_config_os_bsd_netbsd_ctype__base.h
PATCHES="https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-01-13-netbsd-patch1.patch"

# Originally from
# ftp://ftp.netbsd.org/pub/pkgsrc/pkgsrc-2016Q4/pkgsrc/lang/gcc5/patches/patch-libstdc%2B%2B-v3_config_os_bsd_netbsd_ctype__configure__char.cc
PATCHES="$PATCHES https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-01-13-netbsd-patch2.patch"

for patch in $PATCHES; do
  curl $patch | patch -Np0
done

mkdir ../gcc-build
cd ../gcc-build
../gcc-$GCC/configure                            \
  --enable-languages=c,c++                       \
  --target=x86_64-unknown-netbsd                 \
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

cd ../..
rm -rf gcc
