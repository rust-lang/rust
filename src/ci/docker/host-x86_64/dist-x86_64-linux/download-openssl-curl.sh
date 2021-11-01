#!/usr/bin/env bash

set -ex

OPENSSL_VERSION=1.0.2k
CURL_VERSION=7.66.0

curl -f https://ci-mirrors.rust-lang.org/rustc/openssl-$OPENSSL_VERSION.tar.gz -o openssl.tar.gz
curl -f https://ci-mirrors.rust-lang.org/rustc/curl-$CURL_VERSION.tar.xz -o curl.tar.xz
curl -f https://curl.se/ca/cacert.pem -o cacert.pem
