#!/usr/bin/env bash

set -ex
source shared.sh

URL=https://www.kernel.org/pub/software/scm/git/git-2.10.0.tar.gz
SHA256=207cfce8cc0a36497abb66236817ef449a45f6ff9141f586bbe2aafd7bc3d90b

./secure-download.sh $URL $SHA256 | tar xzf -

cd git-2.10.0
make configure
hide_output ./configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf git-2.10.0
