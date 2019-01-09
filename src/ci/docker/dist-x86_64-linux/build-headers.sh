#!/usr/bin/env bash

set -ex
source shared.sh

curl https://cdn.kernel.org/pub/linux/kernel/v3.x/linux-3.2.84.tar.xz | unxz | tar x

cd linux-3.2.84
hide_output make mrproper
hide_output make INSTALL_HDR_PATH=dest headers_install

find dest/include \( -name .install -o -name ..install.cmd \) -delete
yes | cp -fr dest/include/* /usr/include

cd ..
rm -rf linux-3.2.84
