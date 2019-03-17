#!/usr/bin/env bash
# ignore-tidy-linelength

set -ex

wget -O - https://static.redox-os.org/toolchain/x86_64-unknown-redox/relibc-install.tar.gz | \
tar --extract --gzip --directory /usr/local
