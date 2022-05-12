#!/usr/bin/env bash

set -ex

source shared.sh

NINJA=v1.10.2

mkdir ninja
cd ninja

curl -L https://github.com/ninja-build/ninja/archive/refs/tags/$NINJA.tar.gz | \
  tar xzf - --strip-components 1

mkdir ninja-build
cd ninja-build

hide_output cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/rustroot

hide_output make -j$(nproc)
hide_output make install

cd ../..
rm -rf ninja
