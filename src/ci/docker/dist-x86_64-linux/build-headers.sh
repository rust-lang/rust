#!/bin/bash
# Copyright 2017 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

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
