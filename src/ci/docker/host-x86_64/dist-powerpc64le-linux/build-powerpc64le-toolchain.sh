#!/usr/bin/env bash

set -ex

source shared.sh

BINUTILS=2.32
GCC=8.3.0
TARGET=powerpc64le-linux-gnu
SYSROOT=/usr/local/$TARGET/sysroot

# First, download the CentOS7 glibc.ppc64le and relevant header files.
# (upstream ppc64le support wasn't added until 2.19, which el7 backported.)
mkdir -p $SYSROOT
pushd $SYSROOT

# centos_base=http://vault.centos.org/altarch/7.3.1611/os/ppc64le/Packages/
# Mirrored from centos_base above
centos_base=https://ci-mirrors.rust-lang.org/rustc
glibc_v=2.17-157-2020-11-25.el7
kernel_v=3.10.0-514-2020-11-25.el7
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
curl https://ftp.gnu.org/gnu/binutils/binutils-$BINUTILS.tar.xz | tar xJf -
mkdir binutils-build
cd binutils-build
hide_output ../binutils-$BINUTILS/configure --target=$TARGET --with-sysroot=$SYSROOT
hide_output make -j10
hide_output make install
popd
rm -rf binutils-$TARGET

# Finally, download and build gcc.
mkdir gcc-$TARGET
pushd gcc-$TARGET
curl https://ftp.gnu.org/gnu/gcc/gcc-$GCC/gcc-$GCC.tar.xz | tar xJf -
cd gcc-$GCC
hide_output ./contrib/download_prerequisites

mkdir ../gcc-build
cd ../gcc-build
hide_output ../gcc-$GCC/configure                            \
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
hide_output hide_output make -j10
hide_output make install

popd
rm -rf gcc-$TARGET
