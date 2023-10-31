#!/bin/sh
set -ex

URL=https://github.com/crosstool-ng/crosstool-ng
REV=227d99d7f3115f3a078595a580d2b307dcd23e93

mkdir crosstool-ng
cd crosstool-ng
git init
git fetch --depth=1 ${URL} ${REV}
git reset --hard FETCH_HEAD
./bootstrap
./configure --prefix=/usr/local
make -j$(nproc)
make install
cd ..
rm -rf crosstool-ng
