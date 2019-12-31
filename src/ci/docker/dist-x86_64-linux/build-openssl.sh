#!/usr/bin/env bash

set -ex
source shared.sh

VERSION=1.0.2k

# This needs to be downloaded directly from S3, it can't go through the CDN.
# That's because the CDN is backed by CloudFront, which requires SNI and TLSv1
# (without paying an absurd amount of money).
URL=https://rust-lang-ci-mirrors.s3-us-west-1.amazonaws.com/rustc/openssl-$VERSION.tar.gz

curl $URL | tar xzf -

cd openssl-$VERSION
hide_output ./config --prefix=/rustroot shared -fPIC
hide_output make -j10
hide_output make install
cd ..
rm -rf openssl-$VERSION

# Make the system cert collection available to the new install.
ln -nsf /etc/pki/tls/cert.pem /rustroot/ssl/
