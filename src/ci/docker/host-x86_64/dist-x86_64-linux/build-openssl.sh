#!/usr/bin/env bash

set -ex
source shared.sh

tar xzf openssl.tar.gz

cd openssl-*
hide_output ./config --prefix=/rustroot shared -fPIC
hide_output make -j$(nproc)
hide_output make install
cd ..
rm -rf openssl-*

# Make the system cert collection available to the new install.
ln -nsf /etc/pki/tls/cert.pem /rustroot/ssl/
