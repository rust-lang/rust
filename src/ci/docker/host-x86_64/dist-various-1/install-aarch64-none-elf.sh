#!/usr/bin/env bash

set -ex

curl -L https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-elf.tar.xz \
| tar --extract --xz --strip 1 --directory /usr/local
