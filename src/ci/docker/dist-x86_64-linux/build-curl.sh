#!/usr/bin/env bash

set -ex
source shared.sh

VERSION=7.66.0

curl https://rust-lang-ci-mirrors.s3-us-west-1.amazonaws.com/rustc/curl-$VERSION.tar.xz \
  | xz --decompress \
  | tar xf -

mkdir curl-build
cd curl-build
hide_output ../curl-$VERSION/configure \
      --prefix=/rustroot \
      --with-ssl=/rustroot \
      --disable-sspi \
      --disable-gopher \
      --disable-smtp \
      --disable-smb \
      --disable-imap \
      --disable-pop3 \
      --disable-tftp \
      --disable-telnet \
      --disable-manual \
      --disable-dict \
      --disable-rtsp \
      --disable-ldaps \
      --disable-ldap
hide_output make -j10
hide_output make install

cd ..
rm -rf curl-build
rm -rf curl-$VERSION
yum erase -y curl
