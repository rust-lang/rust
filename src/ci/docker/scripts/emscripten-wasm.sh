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

# Download last known good emscripten from WebAssembly waterfall
BUILD=$(curl -fL https://storage.googleapis.com/wasm-llvm/builds/linux/lkgr.json | \
    jq '.build | tonumber')
curl -sL https://storage.googleapis.com/wasm-llvm/builds/linux/$BUILD/wasm-binaries.tbz2 | \
    hide_output tar xvkj

# node 8 is required to run wasm
cd /
curl -sL https://nodejs.org/dist/v8.0.0/node-v8.0.0-linux-x64.tar.xz | \
    tar -xJ

# Make emscripten use wasm-ready node and LLVM tools
echo "EMSCRIPTEN_ROOT = '/wasm-install/emscripten'" >> /root/.emscripten
echo "NODE_JS='/usr/local/bin/node'" >> /root/.emscripten
echo "LLVM_ROOT='/wasm-install/bin'" >> /root/.emscripten
echo "BINARYEN_ROOT = '/wasm-install'" >> /root/.emscripten
echo "COMPILER_ENGINE = NODE_JS" >> /root/.emscripten
echo "JS_ENGINES = [NODE_JS]" >> /root/.emscripten
