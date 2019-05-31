# This script runs `musl-cross-make` to prepare C toolchain (Binutils, GCC, musl itself)
# and builds static libunwind that we distribute for static target.
#
# Versions of the toolchain components are configurable in `musl-cross-make/Makefile` and
# musl unlike GLIBC is forward compatible so upgrading it shouldn't break old distributions.
# Right now we have: Binutils 2.27, GCC 6.4.0, musl 1.1.22.
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

ARCH=$1
TARGET=$ARCH-linux-musl

OUTPUT=/usr/local
shift

# Ancient binutils versions don't understand debug symbols produced by more recent tools.
# Apparently applying `-fPIC` everywhere allows them to link successfully.
export CFLAGS="-fPIC $CFLAGS"

git clone https://github.com/richfelker/musl-cross-make -b v0.9.8
cd musl-cross-make

hide_output make -j$(nproc) TARGET=$TARGET
hide_output make install TARGET=$TARGET OUTPUT=$OUTPUT

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

export CC=$TARGET-gcc
export CXX=$TARGET-g++

LLVM=70

# may have been downloaded in a previous run
if [ ! -d libunwind-release_$LLVM ]; then
  curl -L https://github.com/llvm-mirror/llvm/archive/release_$LLVM.tar.gz | tar xzf -
  curl -L https://github.com/llvm-mirror/libunwind/archive/release_$LLVM.tar.gz | tar xzf -
fi

# fixme(mati865): Replace it with https://github.com/rust-lang/rust/pull/59089
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
