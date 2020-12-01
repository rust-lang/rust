#!/usr/bin/env bash

set -ex

ARCH=x86_64
PATCH_TOOLCHAIN=$1
BINUTILS=2.25.1
GCC=6.4.0

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
  "$@" &> /tmp/build.log
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
  --target=$ARCH-unknown-dragonfly
hide_output make -j10
hide_output make install
cd ../..
rm -rf binutils

# Next, download the DragonFly libc and relevant header files

URL=http://mirror-master.dragonflybsd.org/iso-images/dfly-x86_64-5.0.0_REL.iso.bz2
mkdir dragonfly
curl $URL | bzcat | bsdtar xf - -C dragonfly ./usr/include ./usr/lib ./lib

dst=/usr/local/$ARCH-unknown-dragonfly

mkdir -p $dst/lib
cp -r dragonfly/usr/include $dst/
cp dragonfly/usr/lib/crt1.o $dst/lib
cp dragonfly/usr/lib/Scrt1.o $dst/lib
cp dragonfly/usr/lib/crti.o $dst/lib
cp dragonfly/usr/lib/crtn.o $dst/lib
cp dragonfly/usr/lib/libc.a $dst/lib
cp dragonfly/usr/lib/libutil.a $dst/lib
cp dragonfly/usr/lib/libm.a $dst/lib
cp dragonfly/usr/lib/librt.so.0 $dst/lib
cp dragonfly/usr/lib/libexecinfo.so.1 $dst/lib
cp dragonfly/lib/libc.so.8 $dst/lib
cp dragonfly/lib/libm.so.4 $dst/lib
cp dragonfly/lib/libutil.so.4 $dst/lib
cp dragonfly/usr/lib/libpthread.so $dst/lib/libpthread.so
cp dragonfly/usr/lib/thread/libthread_xu.so.2 $dst/lib/libpthread.so.0

ln -s libc.so.8 $dst/lib/libc.so
ln -s libm.so.4 $dst/lib/libm.so
ln -s librt.so.0 $dst/lib/librt.so
ln -s libutil.so.4 $dst/lib/libutil.so
ln -s libexecinfo.so.1 $dst/lib/libexecinfo.so
rm -rf dragonfly

# Finally, download and build gcc to target DragonFly
mkdir gcc
cd gcc
curl https://ftp.gnu.org/gnu/gcc/gcc-$GCC/gcc-$GCC.tar.gz | tar xzf -
cd gcc-$GCC

# The following three patches are taken from DragonFly's dports collection:
# https://github.com/DragonFlyBSD/DPorts/tree/master/lang/gcc5
# The dports specification for gcc5 contains a few more patches, but they are
# not relevant in this situation, as they are for a language we don't need
# (e.g. java), or a platform which is not supported by DragonFly (e.g. i386,
# powerpc64, ia64, arm).
#
# These patches probably only need to be updated in case the gcc version is
# updated.

patch -p0 < $PATCH_TOOLCHAIN

./contrib/download_prerequisites

mkdir ../gcc-build
cd ../gcc-build
hide_output ../gcc-$GCC/configure                \
  --enable-languages=c,c++                       \
  --target=$ARCH-unknown-dragonfly               \
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
