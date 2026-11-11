#!/bin/sh
# This script runs `musl-cross-make` to prepare C toolchain (Binutils, GCC, musl itself)
# and builds static libunwind that we distribute for static target.
#
# Versions of the toolchain components are configurable in `musl-cross-make/Makefile` and
# musl unlike GLIBC is forward compatible so upgrading it shouldn't break old distributions.
# Right now we have: Binutils 2.44, GCC 9.4.0, musl 1.2.6.

# ignore-tidy-file-linelength

set -ex

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
  rm /tmp/build.log
  set -x
}

ARCH=$1
TARGET=$ARCH-linux-musl

# Don't depend on the mirrors of sabotage linux that musl-cross-make uses.
LINUX_HEADERS_SITE=https://ci-mirrors.rust-lang.org/rustc/sabotage-linux-tarballs
LINUX_VER=headers-4.19.88

OUTPUT=/usr/local
shift

# Ancient binutils versions don't understand debug symbols produced by more recent tools.
# Apparently applying `-fPIC` everywhere allows them to link successfully.
# Enable debug info. If we don't do so, users can't debug into musl code,
# debuggers can't walk the stack, etc. Fixes #90103.
export CFLAGS="-fPIC -g1 $CFLAGS"

git clone https://github.com/richfelker/musl-cross-make # -b v0.9.9
cd musl-cross-make
# A version that includes musl 1.2.6 and patches for CVE-2026-6042,
# CVE-2026-40200, and CVE-2026-107891.
git checkout 5e8042e63de29f8473fc3b0b831a587ba284e1a8

hide_output make -j$(nproc) TARGET=$TARGET MUSL_VER=1.2.6 LINUX_HEADERS_SITE=$LINUX_HEADERS_SITE LINUX_VER=$LINUX_VER
hide_output make install TARGET=$TARGET MUSL_VER=1.2.6 LINUX_HEADERS_SITE=$LINUX_HEADERS_SITE LINUX_VER=$LINUX_VER OUTPUT=$OUTPUT

cd -

# Install musl library to make binaries executable
ln -s $OUTPUT/$TARGET/lib/libc.so /lib/ld-musl-$ARCH.so.1
echo $OUTPUT/$TARGET/lib >> /etc/ld-musl-$ARCH.path

# Now when musl bootstraps itself create proper toolchain symlinks to make build and tests easier
if [ "$REPLACE_CC" = "1" ]; then
    for exec in cc gcc; do
        ln -s $TARGET-gcc /usr/local/bin/$exec
    done
    for exec in cpp c++ g++; do
        ln -s $TARGET-g++ /usr/local/bin/$exec
    done
fi
