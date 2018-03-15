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

curl https://ftp.gnu.org/gnu/binutils/binutils-2.25.1.tar.bz2 | tar xfj -

mkdir binutils-build
cd binutils-build
hide_output ../binutils-2.25.1/configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf binutils-build
rm -rf binutils-2.25.1
