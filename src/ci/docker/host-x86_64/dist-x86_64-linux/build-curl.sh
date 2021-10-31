#!/usr/bin/env bash

set -ex
source shared.sh

tar xJf curl.tar.xz

mkdir curl-build
cd curl-build
hide_output ../curl-*/configure \
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
hide_output make -j$(nproc)
hide_output make install

cd ..
rm -rf curl-build
rm -rf curl-*
