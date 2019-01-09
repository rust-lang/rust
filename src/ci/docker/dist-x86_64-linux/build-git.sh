#!/usr/bin/env bash

set -ex
source shared.sh

curl -L https://www.kernel.org/pub/software/scm/git/git-2.10.0.tar.gz | tar xzf -

cd git-2.10.0
make configure
hide_output ./configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf git-2.10.0
