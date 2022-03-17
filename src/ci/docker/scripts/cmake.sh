#!/bin/bash
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

# LLVM 12 requires CMake 3.13.4 or higher.
# This script is not necessary for images using Ubuntu 20.04 or newer.
CMAKE=3.13.4
curl -L https://github.com/Kitware/CMake/releases/download/v$CMAKE/cmake-$CMAKE.tar.gz | tar xzf -

mkdir cmake-build
cd cmake-build
ls -al /tmp/build.log || true
who am i
../cmake-$CMAKE/configure
make -j$(nproc)
make install

cd ..
rm -rf cmake-build
rm -rf cmake-$CMAKE
