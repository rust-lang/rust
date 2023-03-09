#!/usr/bin/env bash

set -ex
source shared.sh

ARCH=$1
LIB_ARCH=$2
APT_ARCH=$3
MANUFACTURER=$4
BINUTILS=2.28.1
GCC=6.5.0

TARGET=${ARCH}-${MANUFACTURER}-solaris2.10

# First up, build binutils
mkdir binutils
cd binutils

curl https://ftp.gnu.org/gnu/binutils/binutils-$BINUTILS.tar.xz | tar xJf -
mkdir binutils-build
cd binutils-build
hide_output ../binutils-$BINUTILS/configure --target=$TARGET
hide_output make -j10
hide_output make install

cd ../..
rm -rf binutils

# Next, download and install the relevant solaris packages
mkdir solaris
cd solaris

dpkg --add-architecture $APT_ARCH
apt-get update
apt-get install -y --download-only                           \
  libc:$APT_ARCH                                             \
  liblgrp:$APT_ARCH                                          \
  libm-dev:$APT_ARCH                                         \
  libpthread:$APT_ARCH                                       \
  libresolv:$APT_ARCH                                        \
  librt:$APT_ARCH                                            \
  libsendfile:$APT_ARCH                                      \
  libsocket:$APT_ARCH                                        \
  system-crt:$APT_ARCH                                       \
  system-header:$APT_ARCH

for deb in /var/cache/apt/archives/*$APT_ARCH.deb; do
  dpkg -x $deb .
done
apt-get clean

# The -dev packages are not available from the apt repository we're using.
# However, those packages are just symlinks from *.so to *.so.<version>.
# This makes all those symlinks.
for lib in $(find -name '*.so.*'); do
  target=${lib%.so.*}.so
  [ -e $target ] || ln -s ${lib##*/} $target
done

# Remove Solaris 11 functions that are optionally used by libbacktrace.
# This is for Solaris 10 compatibility.
rm usr/include/link.h
patch -p0  << 'EOF'
--- usr/include/string.h
+++ usr/include/string10.h
@@ -93 +92,0 @@
-extern size_t strnlen(const char *, size_t);
EOF

mkdir                  /usr/local/$TARGET/usr
mv usr/include         /usr/local/$TARGET/usr/include
mv usr/lib/$LIB_ARCH/* /usr/local/$TARGET/lib
mv     lib/$LIB_ARCH/* /usr/local/$TARGET/lib

ln -s usr/include /usr/local/$TARGET/sys-include
ln -s usr/include /usr/local/$TARGET/include

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
  --target=$TARGET                \
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
  --disable-lto

hide_output make -j10
hide_output make install

cd ../..
rm -rf gcc
