#!/usr/bin/env bash

set -ex
source shared.sh

VERSION=2.21.0
URL=https://www.kernel.org/pub/software/scm/git/git-$VERSION.tar.gz
SHA256=85eca51c7404da75e353eba587f87fea9481ba41e162206a6f70ad8118147bee

./secure-download.sh $URL $SHA256 | tar xzf -

cd git-$VERSION
make configure
hide_output ./configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf git-$VERSION
