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

curl https://ftp.gnu.org/gnu/gcc/gcc-4.7.4/gcc-4.7.4.tar.bz2 | tar xjf -
cd gcc-4.7.4
./contrib/download_prerequisites
mkdir ../gcc-build
cd ../gcc-build
hide_output ../gcc-4.7.4/configure \
    --prefix=/rustroot \
    --enable-languages=c,c++
hide_output make -j10
hide_output make install
ln -nsf gcc /rustroot/bin/cc

cd ..
rm -rf gcc-build
rm -rf gcc-4.7.4
yum erase -y gcc binutils
