#!/bin/sh
set -ex

# ignore-tidy-file-linelength

URL=https://github.com/crosstool-ng/crosstool-ng
REV=27cd8380e72bb1cf3e7cf4a06a9cdbdc57df6f72

mkdir crosstool-ng
cd crosstool-ng
git init
git fetch --depth=1 ${URL} ${REV}
git reset --hard FETCH_HEAD

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
rm -rf crosstool-ng
