#!/usr/bin/env bash

set -ex

curl https://static.redox-os.org/toolchain/x86_64-unknown-redox/relibc-install.tar.gz | \
tar --extract --gzip --directory /usr/local
