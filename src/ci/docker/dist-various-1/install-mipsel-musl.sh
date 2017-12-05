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

mkdir /usr/local/mipsel-linux-musl

# Note that this originally came from:
# https://downloads.openwrt.org/snapshots/trunk/malta/generic/
# OpenWrt-Toolchain-malta-le_gcc-5.3.0_musl-1.1.15.Linux-x86_64.tar.bz2
URL="https://s3-us-west-1.amazonaws.com/rust-lang-ci2/libc"
FILE="OpenWrt-Toolchain-malta-le_gcc-5.3.0_musl-1.1.15.Linux-x86_64.tar.bz2"
curl -L "$URL/$FILE" | tar xjf - -C /usr/local/mipsel-linux-musl --strip-components=2

for file in /usr/local/mipsel-linux-musl/bin/mipsel-openwrt-linux-*; do
  ln -s $file /usr/local/bin/`basename $file`
done
