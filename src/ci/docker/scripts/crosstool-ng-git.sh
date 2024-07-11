#!/bin/sh
set -ex

URL=https://github.com/crosstool-ng/crosstool-ng
REV=c64500d94be92ed1bcdfdef911048a14e216a5e1

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
