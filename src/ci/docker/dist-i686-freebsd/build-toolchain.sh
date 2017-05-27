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

ARCH=$1
BINUTILS=2.25.1
GCC=5.3.0

hide_output() {
  set +x
  on_err="
echo ERROR: An error was encountered with the build.
cat /tmp/build.log
exit 1
"
  trap "$on_err" ERR
  bash -c "while true; do sleep 30; echo \$(date) - building ...; done" &
  PING_LOOP_PID=$!
  $@ &> /tmp/build.log
  trap - ERR
  kill $PING_LOOP_PID
  set -x
}

mkdir binutils
cd binutils

# First up, build binutils
curl https://ftp.gnu.org/gnu/binutils/binutils-$BINUTILS.tar.bz2 | tar xjf -
mkdir binutils-build
cd binutils-build
hide_output ../binutils-$BINUTILS/configure \
  --target=$ARCH-unknown-freebsd10
hide_output make -j10
hide_output make install
cd ../..
rm -rf binutils

# Next, download the FreeBSD libc and relevant header files

mkdir freebsd
case "$ARCH" in
    x86_64)
        URL=ftp://ftp.freebsd.org/pub/FreeBSD/releases/amd64/10.2-RELEASE/base.txz
        ;;
    i686)
        URL=ftp://ftp.freebsd.org/pub/FreeBSD/releases/i386/10.2-RELEASE/base.txz
        ;;
esac
curl $URL | tar xJf - -C freebsd ./usr/include ./usr/lib ./lib

dst=/usr/local/$ARCH-unknown-freebsd10

cp -r freebsd/usr/include $dst/
cp freebsd/usr/lib/crt1.o $dst/lib
cp freebsd/usr/lib/Scrt1.o $dst/lib
cp freebsd/usr/lib/crti.o $dst/lib
cp freebsd/usr/lib/crtn.o $dst/lib
cp freebsd/usr/lib/libc.a $dst/lib
cp freebsd/usr/lib/libutil.a $dst/lib
cp freebsd/usr/lib/libutil_p.a $dst/lib
cp freebsd/usr/lib/libm.a $dst/lib
cp freebsd/usr/lib/librt.so.1 $dst/lib
cp freebsd/usr/lib/libexecinfo.so.1 $dst/lib
cp freebsd/lib/libc.so.7 $dst/lib
cp freebsd/lib/libm.so.5 $dst/lib
cp freebsd/lib/libutil.so.9 $dst/lib
cp freebsd/lib/libthr.so.3 $dst/lib/libpthread.so

ln -s libc.so.7 $dst/lib/libc.so
ln -s libm.so.5 $dst/lib/libm.so
ln -s librt.so.1 $dst/lib/librt.so
ln -s libutil.so.9 $dst/lib/libutil.so
ln -s libexecinfo.so.1 $dst/lib/libexecinfo.so
rm -rf freebsd

# Finally, download and build gcc to target FreeBSD
mkdir gcc
cd gcc
curl https://ftp.gnu.org/gnu/gcc/gcc-$GCC/gcc-$GCC.tar.bz2 | tar xjf -
cd gcc-$GCC
./contrib/download_prerequisites

mkdir ../gcc-build
cd ../gcc-build
hide_output ../gcc-$GCC/configure                \
  --enable-languages=c,c++                       \
  --target=$ARCH-unknown-freebsd10               \
  --disable-multilib                             \
  --disable-nls                                  \
  --disable-libgomp                              \
  --disable-libquadmath                          \
  --disable-libssp                               \
  --disable-libvtv                               \
  --disable-libcilkrts                           \
  --disable-libada                               \
  --disable-libsanitizer                         \
  --disable-libquadmath-support                  \
  --disable-lto
hide_output make -j10
hide_output make install
cd ../..
rm -rf gcc
