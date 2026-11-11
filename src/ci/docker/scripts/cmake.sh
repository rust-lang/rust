#!/bin/bash
set -ex

hide_output() {
  set +x
  on_err="
echo ERROR: An error was encountered with the build.
cat /tmp/cmake_build.log
exit 1
"
  trap "$on_err" ERR
  bash -c "while true; do sleep 30; echo \$(date) - building ...; done" &
  PING_LOOP_PID=$!
  "$@" &> /tmp/cmake_build.log
  trap - ERR
  kill $PING_LOOP_PID
  rm /tmp/cmake_build.log
  set -x
}

# LLVM 17 requires CMake 3.20 or higher.
# This script is not necessary for images using Ubuntu 22.04 or newer.
CMAKE=3.20.3
curl -L https://github.com/Kitware/CMake/releases/download/v$CMAKE/cmake-$CMAKE.tar.gz | tar xzf -

mkdir cmake-build
cd cmake-build
hide_output ../cmake-$CMAKE/configure
hide_output make -j$(nproc)
hide_output make install

cd ..
rm -rf cmake-build
rm -rf cmake-$CMAKE
