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
  trap - ERR
  kill $PING_LOOP_PID
  rm /tmp/build.log
  set -x
}

git clone https://github.com/rumpkernel/rumprun
cd rumprun
git reset --hard b04d42225a12a6fae57a78a9c1cf23642e46cd00
git submodule update --init

# Disable -Werror, to avoid breaking the build with newer compilers.
CC=cc NOGCCERROR=1 hide_output ./build-rr.sh -d /usr/local hw
cd ..
rm -rf rumprun
