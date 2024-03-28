#!/usr/bin/env bash

set -ex

archive=llvm-mingw-20231128-ucrt-ubuntu-20.04-x86_64.tar.xz
curl -L https://github.com/mstorsjo/llvm-mingw/releases/download/20231128/${archive} | \
tar --extract --lzma --strip 1 --directory /usr/local
