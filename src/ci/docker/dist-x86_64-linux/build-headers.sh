#!/usr/bin/env bash

set -ex
source shared.sh

URL=https://cdn.kernel.org/pub/linux/kernel/v3.x/linux-3.2.84.tar.xz
SHA256=66e329b56487a88f07274a4fa8ec1dfdab8a3df5c3812dd03589d393941b1d47

./secure-download.sh $URL $SHA256 | unxz | tar x

cd linux-3.2.84
hide_output make mrproper
hide_output make INSTALL_HDR_PATH=dest headers_install

find dest/include \( -name .install -o -name ..install.cmd \) -delete
yes | cp -fr dest/include/* /usr/include

cd ..
rm -rf linux-3.2.84
