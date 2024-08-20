#!/bin/sh
set -ex

URL=https://github.com/crosstool-ng/crosstool-ng
REV=ed12fa68402f58e171a6f79500f73f4781fdc9e5

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
