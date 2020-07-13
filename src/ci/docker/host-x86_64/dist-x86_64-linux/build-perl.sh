#!/usr/bin/env bash

set -ex
source shared.sh

curl https://www.cpan.org/src/5.0/perl-5.28.0.tar.gz | \
  tar xzf -

cd perl-5.28.0

# Gotta do some hackery to tell python about our custom OpenSSL build, but other
# than that fairly normal.
CC=gcc \
CFLAGS='-I /rustroot/include -fgnu89-inline' \
LDFLAGS='-L /rustroot/lib -L /rustroot/lib64' \
    hide_output ./configure.gnu
hide_output make -j10
hide_output make install

cd ..
rm -rf perl-5.28.0
