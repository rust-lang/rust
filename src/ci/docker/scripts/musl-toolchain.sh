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
  rm /tmp/build.log
  set -x
}

TARGET=$1
#ARCH=$1
#TARGET=linux-musl-$ARCH
ARCH=x86_64

OUTPUT=/usr/local
shift

git clone https://github.com/richfelker/musl-cross-make -b v0.9.7
cd musl-cross-make

hide_output make -j$(nproc) TARGET=$TARGET
hide_output make install TARGET=$TARGET OUTPUT=$OUTPUT

cd -

# Make musl binaries executable

ln -s $OUTPUT/$TARGET/lib/libc.so /lib/ld-musl-$ARCH.so.1
echo $OUTPUT/$TARGET/lib >> /etc/ld-musl-$ARCH.path


export CC=$TARGET-gcc
export CXX=$TARGET-g++

LLVM=60

# may have been downloaded in a previous run
if [ ! -d libunwind-release_$LLVM ]; then
  curl -L https://github.com/llvm-mirror/llvm/archive/release_$LLVM.tar.gz | tar xzf -
  curl -L https://github.com/llvm-mirror/libunwind/archive/release_$LLVM.tar.gz | tar xzf -
fi

mkdir libunwind-build
cd libunwind-build
cmake ../libunwind-release_$LLVM \
          -DLLVM_PATH=/build/llvm-release_$LLVM \
          -DLIBUNWIND_ENABLE_SHARED=0 \
          -DCMAKE_C_COMPILER=$CC \
          -DCMAKE_CXX_COMPILER=$CXX \
          -DCMAKE_C_FLAGS="$CFLAGS" \
          -DCMAKE_CXX_FLAGS="$CXXFLAGS"

hide_output make -j$(nproc)
cp lib/libunwind.a $OUTPUT/$TARGET/lib
cd - && rm -rf libunwind-build

