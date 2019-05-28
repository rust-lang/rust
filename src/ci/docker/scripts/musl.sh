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

TAG=$1
shift

export CFLAGS="-fPIC $CFLAGS"

MUSL=musl-1.1.22

# may have been downloaded in a previous run
if [ ! -d $MUSL ]; then
  curl https://www.musl-libc.org/releases/$MUSL.tar.gz | tar xzf -
fi

cd $MUSL
./configure --enable-optimize --enable-debug --disable-shared --prefix=/musl-$TAG $@
if [ "$TAG" = "i586" -o "$TAG" = "i686" ]; then
  hide_output make -j$(nproc) AR=ar RANLIB=ranlib
else
  hide_output make -j$(nproc)
fi
hide_output make install
hide_output make clean

cd ..

LLVM=70

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
cp lib/libunwind.a /musl-$TAG/lib
cd ../ && rm -rf libunwind-build
