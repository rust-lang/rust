#!/bin/sh
set -ex

CT_NG=1.27.0

url="https://github.com/crosstool-ng/crosstool-ng/archive/crosstool-ng-$CT_NG.tar.gz"
curl -Lf $url | tar xzf -
cd crosstool-ng-crosstool-ng-$CT_NG

# https://github.com/crosstool-ng/crosstool-ng/issues/1832
# "download source of zlib is invalid now"
sed -e "s|zlib.net/'|zlib.net/fossils'|" -i packages/zlib/package.desc

./bootstrap
./configure --prefix=/usr/local
make -j$(nproc)
make install
cd ..
rm -rf crosstool-ng-crosstool-ng-$CT_NG
