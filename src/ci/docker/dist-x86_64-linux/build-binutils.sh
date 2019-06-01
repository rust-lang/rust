#!/usr/bin/env bash

set -ex

source shared.sh

URL=https://ftp.gnu.org/gnu/binutils/binutils-2.25.1.tar.bz2
SHA256=b5b14added7d78a8d1ca70b5cb75fef57ce2197264f4f5835326b0df22ac9f22

./secure-download.sh $URL $SHA256 | tar xfj -

mkdir binutils-build
cd binutils-build
hide_output ../binutils-2.25.1/configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf binutils-build
rm -rf binutils-2.25.1
