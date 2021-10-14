#!/usr/bin/env bash

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
  rm /tmp/build.log
  trap - ERR
  kill $PING_LOOP_PID
  set -x
}

mkdir build
cd build
cp ../arm-linux-gnueabihf.config .config
# FIXME ct-ng oldconfig is not working as intended.
# ct-ng oldconfig
hide_output ct-ng build
cd ..
rm -rf build
