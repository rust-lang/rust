#!/bin/sh
set -ex

# ignore-tidy-file-linelength

CT_NG=1.28.0

url="https://github.com/crosstool-ng/crosstool-ng/archive/crosstool-ng-$CT_NG.tar.gz"
curl -Lf $url | tar xzf -
cd crosstool-ng-crosstool-ng-$CT_NG

# https://github.com/crosstool-ng/crosstool-ng/issues/1832
# "download source of zlib is invalid now"
sed -e "s|zlib.net/'|zlib.net/fossils'|" -i packages/zlib/package.desc

# FIXME(#158718): patch crosstools-ng known-good kernel artifact SHA256
# checksums to the artifacts we mirror in `ci-mirrors`.
# See
# <https://github.com/rust-lang/ci-mirrors/blob/b474b4bb35108dab668907172c858854f209c809/files/rustc/kernel.toml>.
patch -p1 </scripts/crosstool-ng-sha256-20260705.diff

./bootstrap
./configure --prefix=/usr/local
make -j$(nproc)
make install
cd ..
rm -rf crosstool-ng-crosstool-ng-$CT_NG
