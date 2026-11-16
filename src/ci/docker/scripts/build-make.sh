#!/bin/sh
set -ex

MAKE_VERSION="4.4.1"
curl -f "https://ci-mirrors.rust-lang.org/rustc/gcc/make-${MAKE_VERSION}.tar.gz" | tar xzf -
cd "make-${MAKE_VERSION}"
./configure --prefix=/rustroot
make
make install
cd ..
rm -rf "make-${MAKE_VERSION}"
