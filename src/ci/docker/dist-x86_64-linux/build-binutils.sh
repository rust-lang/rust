#!/usr/bin/env bash

set -ex

source shared.sh

curl https://ftp.gnu.org/gnu/binutils/binutils-2.25.1.tar.bz2 | tar xfj -

mkdir binutils-build
cd binutils-build
hide_output ../binutils-2.25.1/configure --prefix=/rustroot
hide_output make -j10
hide_output make install

cd ..
rm -rf binutils-build
rm -rf binutils-2.25.1
