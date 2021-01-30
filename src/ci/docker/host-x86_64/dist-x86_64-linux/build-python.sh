#!/usr/bin/env bash

set -ex
source shared.sh

curl https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz | \
  tar xzf -

mkdir python-build
cd python-build

# Gotta do some hackery to tell python about our custom OpenSSL build, but other
# than that fairly normal.
CFLAGS='-I /rustroot/include' LDFLAGS='-L /rustroot/lib -L /rustroot/lib64' \
    hide_output ../Python-3.9.1/configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf python-build
rm -rf Python-3.9.1
