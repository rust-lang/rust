#!/bin/bash
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
  rm /tmp/build.log
  trap - ERR
  kill $PING_LOOP_PID
  set -x
}

mkdir build
cd build
cp ../armv7-linux-gnueabihf.config .config
ct-ng oldconfig
hide_output ct-ng build
cd ..
rm -rf build

mkdir build
cd build
cp ../aarch64-linux-gnu.config .config
ct-ng oldconfig
hide_output ct-ng build
cd ..
rm -rf build
