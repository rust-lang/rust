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

git clone https://github.com/emscripten-core/emsdk.git /emsdk-portable
cd /emsdk-portable
hide_output ./emsdk install 1.38.46-upstream
./emsdk activate 1.38.46-upstream

# Compile and cache libc
source ./emsdk_env.sh
echo "main(){}" > a.c
HOME=/emsdk-portable/ emcc a.c
rm -f a.*

# Make emsdk usable by any user
cp /root/.emscripten /emsdk-portable
chmod a+rxw -R /emsdk-portable
