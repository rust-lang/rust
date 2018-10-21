#!/usr/bin/env bash
# Copyright 2018 The Rust Project Developers. See the COPYRIGHT
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

curl https://www.cpan.org/src/5.0/perl-5.28.0.tar.gz | \
  tar xzf -

cd perl-5.28.0

# Gotta do some hackery to tell python about our custom OpenSSL build, but other
# than that fairly normal.
CC=gcc \
CFLAGS='-I /rustroot/include' LDFLAGS='-L /rustroot/lib -L /rustroot/lib64' \
    hide_output ./configure.gnu
hide_output make -j10
hide_output make install

cd ..
rm -rf perl-5.28.0
