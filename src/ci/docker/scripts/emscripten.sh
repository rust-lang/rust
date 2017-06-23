# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
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
  rm -f /tmp/build.log
  set -x
}

# Download emsdk
cd /
curl -L https://s3.amazonaws.com/mozilla-games/emscripten/releases/emsdk-portable.tar.gz | \
    tar -xz

# Download last known good emscripten from WebAssembly waterfall
BUILD=$(curl -L https://storage.googleapis.com/wasm-llvm/builds/linux/lkgr.json | \
    jq '.build | tonumber')
curl -L https://storage.googleapis.com/wasm-llvm/builds/linux/$BUILD/wasm-binaries.tbz2 | \
    hide_output tar xvkj

# node 8 is required to run wasm
cd /
curl -L https://nodejs.org/dist/v8.0.0/node-v8.0.0-linux-x64.tar.xz | \
    tar -xJ

cd /emsdk-portable
./emsdk update
hide_output ./emsdk install sdk-1.37.13-64bit
./emsdk activate sdk-1.37.13-64bit

# Make emscripten use wasm-ready node and LLVM tools
echo "NODE_JS='/node-v8.0.0-linux-x64/bin/node'" >> /root/.emscripten
echo "LLVM_ROOT='/wasm-install/bin'" >> /root/.emscripten

# Make emsdk usable by any user
cp /root/.emscripten /emsdk-portable
chmod a+rxw -R /emsdk-portable

# Compile and cache libc
source ./emsdk_env.sh
echo "main(){}" > a.c
HOME=/emsdk-portable/ emcc a.c
HOME=/emsdk-portable/ emcc -s WASM=1 a.c
rm -f a.*
