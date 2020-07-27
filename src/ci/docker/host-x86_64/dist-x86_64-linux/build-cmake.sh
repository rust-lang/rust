#!/usr/bin/env bash

set -ex
source shared.sh

CMAKE=3.13.4
curl -L https://github.com/Kitware/CMake/releases/download/v$CMAKE/cmake-$CMAKE.tar.gz | tar xzf -

mkdir cmake-build
cd cmake-build
hide_output ../cmake-$CMAKE/configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf cmake-build
rm -rf cmake-$CMAKE
