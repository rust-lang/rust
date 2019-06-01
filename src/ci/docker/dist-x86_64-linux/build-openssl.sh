#!/usr/bin/env bash

set -ex
source shared.sh

VERSION=1.0.2k
URL=https://rust-lang-ci2.s3.amazonaws.com/rust-ci-mirror/openssl-$VERSION.tar.gz
SHA256=6b3977c61f2aedf0f96367dcfb5c6e578cf37e7b8d913b4ecb6643c3cb88d8c0

./secure-download.sh $URL $SHA256 | tar xzf -

cd openssl-$VERSION
hide_output ./config --prefix=/rustroot shared -fPIC
hide_output make -j10
hide_output make install
cd ..
rm -rf openssl-$VERSION

# Make the system cert collection available to the new install.
ln -nsf /etc/pki/tls/cert.pem /rustroot/ssl/
