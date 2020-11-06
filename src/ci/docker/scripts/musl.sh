#!/bin/sh
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

TAG=$1
shift

# Ancient binutils versions don't understand debug symbols produced by more recent tools.
# Apparently applying `-fPIC` everywhere allows them to link successfully.
export CFLAGS="-fPIC $CFLAGS"

MUSL=musl-1.1.24

# may have been downloaded in a previous run
if [ ! -d $MUSL ]; then
  curl https://www.musl-libc.org/releases/$MUSL.tar.gz | tar xzf -
fi

cd $MUSL
./configure --enable-optimize --enable-debug --disable-shared --prefix=/musl-$TAG "$@"
if [ "$TAG" = "i586" -o "$TAG" = "i686" ]; then
  hide_output make -j$(nproc) AR=ar RANLIB=ranlib
else
  hide_output make -j$(nproc)
fi
hide_output make install
hide_output make clean
