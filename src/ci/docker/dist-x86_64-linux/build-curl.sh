#!/usr/bin/env bash

set -ex
source shared.sh

VERSION=7.51.0

curl http://cool.haxx.se/download/curl-$VERSION.tar.bz2 | tar xjf -

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
