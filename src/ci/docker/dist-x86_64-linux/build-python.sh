#!/usr/bin/env bash

set -ex
source shared.sh

URL=https://www.python.org/ftp/python/2.7.12/Python-2.7.12.tgz
SHA256=3cb522d17463dfa69a155ab18cffa399b358c966c0363d6c8b5b3bf1384da4b6

./secure-download.sh $URL $SHA256 | tar xzf -

mkdir python-build
cd python-build

# Gotta do some hackery to tell python about our custom OpenSSL build, but other
# than that fairly normal.
CFLAGS='-I /rustroot/include' LDFLAGS='-L /rustroot/lib -L /rustroot/lib64' \
    hide_output ../Python-2.7.12/configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf python-build
rm -rf Python-2.7.12
