#!/usr/bin/env bash

set -ex

release_date=20250430
archive=llvm-mingw-${release_date}-ucrt-ubuntu-22.04-x86_64.tar.xz
curl -L https://github.com/mstorsjo/llvm-mingw/releases/download/${release_date}/${archive} | \
tar --extract --xz --strip 1 --directory /usr/local

# https://github.com/mstorsjo/llvm-mingw/issues/493
ln -s aarch64-w64-windows-gnu.cfg /usr/local/bin/aarch64-pc-windows-gnu.cfg
