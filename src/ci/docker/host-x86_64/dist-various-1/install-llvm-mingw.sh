#!/usr/bin/env bash

set -ex

curl -L https://github.com/mstorsjo/llvm-mingw/releases/download/20230614/llvm-mingw-20230614-ucrt-ubuntu-20.04-x86_64.tar.xz | \
tar --extract --lzma --strip 1 --directory /usr/local
