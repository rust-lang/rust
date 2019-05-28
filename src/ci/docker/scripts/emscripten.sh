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

cd /
curl -fL https://mozilla-games.s3.amazonaws.com/emscripten/releases/emsdk-portable.tar.gz | \
    tar -xz

cd /emsdk-portable
./emsdk update
hide_output ./emsdk install sdk-1.38.15-64bit
./emsdk activate sdk-1.38.15-64bit

# Compile and cache libc
source ./emsdk_env.sh
echo "main(){}" > a.c
HOME=/emsdk-portable/ emcc a.c
HOME=/emsdk-portable/ emcc -s BINARYEN=1 a.c
rm -f a.*

# Make emsdk usable by any user
cp /root/.emscripten /emsdk-portable
chmod a+rxw -R /emsdk-portable
