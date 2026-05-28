#!/bin/bash
set -ex

hide_output() {
  set +x
  on_err="
echo ERROR: An error was encountered with the build.
cat /tmp/zstd_build.log
exit 1
"
  trap "$on_err" ERR
  bash -c "while true; do sleep 30; echo \$(date) - building ...; done" &
  PING_LOOP_PID=$!
  "$@" &> /tmp/zstd_build.log
  trap - ERR
  kill $PING_LOOP_PID
  rm /tmp/zstd_build.log
  set -x
}

ZSTD=1.5.6
curl -L https://github.com/facebook/zstd/releases/download/v$ZSTD/zstd-$ZSTD.tar.gz | tar xzf -

cd zstd-$ZSTD
CFLAGS=-fPIC hide_output make -j$(nproc) VERBOSE=1
hide_output make install

# It doesn't seem to be possible to move destination directory
# of the `make install` above. We thus copy the built artifacts
# manually to our custom rustroot, so that it can be found through
# LD_LIBRARY_PATH.
cp /usr/local/lib/libzstd* /rustroot/lib64

cd ..
rm -rf zstd-$ZSTD
