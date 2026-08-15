#!/bin/sh
set -ex

URL=https://github.com/ohos-rs/ohos-openssl/archive/refs/tags/0.1.0.tar.gz

mkdir -p /opt/ohos-openssl
curl -fL $URL | tar xz -C /opt/ohos-openssl --strip-components=1
