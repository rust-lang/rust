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

curl https://www.kernel.org/pub/software/scm/git/git-2.10.0.tar.gz | tar xzf -

cd git-2.10.0
make configure
hide_output ./configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf git-2.10.0
