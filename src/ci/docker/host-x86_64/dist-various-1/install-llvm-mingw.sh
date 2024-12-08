#!/usr/bin/env bash

set -ex

release_date=20240404
archive=llvm-mingw-${release_date}-ucrt-ubuntu-20.04-x86_64.tar.xz
curl -L https://github.com/mstorsjo/llvm-mingw/releases/download/${release_date}/${archive} | \
tar --extract --lzma --strip 1 --directory /usr/local
