#!/usr/bin/env bash

set -ex
source shared.sh

URL=https://cmake.org/files/v3.6/cmake-3.6.3.tar.gz
SHA256=7d73ee4fae572eb2d7cd3feb48971aea903bb30a20ea5ae8b4da826d8ccad5fe

./secure-download.sh $URL $SHA256 | tar xzf -

mkdir cmake-build
cd cmake-build
hide_output ../cmake-3.6.3/configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf cmake-build
rm -rf cmake-3.6.3
