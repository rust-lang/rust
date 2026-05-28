#!/bin/sh
# This script runs `musl-cross-make` to prepare C toolchain (Binutils, GCC, musl itself)
# and builds static libunwind that we distribute for static target.
#
# Versions of the toolchain components are configurable in `musl-cross-make/Makefile` and
# musl unlike GLIBC is forward compatible so upgrading it shouldn't break old distributions.
# Right now we have: Binutils 2.31.1, GCC 9.2.0, musl 1.2.5.

# ignore-tidy-linelength

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
# A version that includes support for building musl 1.2.5
git checkout 3635262e4524c991552789af6f36211a335a77b3

# Patch CVE-2026-6042: https://www.openwall.com/lists/oss-security/2026/04/09/19
# Patch CVE-2026-40200: https://www.openwall.com/lists/musl/2026/04/10/3
# These should be removed when musl-cross-make adds them, or we upgrade to musl >= 1.2.7.
cp /build/musl-cve-2026-6042.diff ./patches/musl-1.2.5/0003-cve-2026-6042.diff
cp /build/musl-cve-2026-40200.diff ./patches/musl-1.2.5/0004-cve-2026-40200.diff

hide_output make -j$(nproc) TARGET=$TARGET MUSL_VER=1.2.5 LINUX_HEADERS_SITE=$LINUX_HEADERS_SITE LINUX_VER=$LINUX_VER
hide_output make install TARGET=$TARGET MUSL_VER=1.2.5 LINUX_HEADERS_SITE=$LINUX_HEADERS_SITE LINUX_VER=$LINUX_VER OUTPUT=$OUTPUT

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
