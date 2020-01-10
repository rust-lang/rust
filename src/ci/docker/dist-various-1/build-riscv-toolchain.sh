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
  $@ &> /tmp/build.log
  rm /tmp/build.log
  trap - ERR
  kill $PING_LOOP_PID
  set -x
}

mkdir -p /tmp/build-riscv
cp riscv64-unknown-linux-gnu.config /tmp/build-riscv/.config
cd /tmp/build-riscv
hide_output ct-ng build
cd ..
rm -rf build-riscv
