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

VERSION=1.0.2j

curl https://www.openssl.org/source/openssl-$VERSION.tar.gz | tar xzf -

cd openssl-$VERSION
hide_output ./config --prefix=/rustroot shared -fPIC
hide_output make -j10
hide_output make install
cd ..
rm -rf openssl-$VERSION

# Make the system cert collection available to the new install.
ln -nsf /etc/pki/tls/cert.pem /rustroot/ssl/
