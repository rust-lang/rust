#!/usr/bin/env bash

set -ex
source shared.sh

curl https://cmake.org/files/v3.6/cmake-3.6.3.tar.gz | tar xzf -

mkdir cmake-build
cd cmake-build
hide_output ../cmake-3.6.3/configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf cmake-build
rm -rf cmake-3.6.3
