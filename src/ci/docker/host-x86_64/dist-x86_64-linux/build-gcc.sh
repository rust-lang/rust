#!/usr/bin/env bash
set -ex

source shared.sh

GCC=5.5.0

curl https://ftp.gnu.org/gnu/gcc/gcc-$GCC/gcc-$GCC.tar.xz | xzcat | tar xf -
cd gcc-$GCC

# FIXME(#49246): Remove the `sed` below.
#
# On 2018 March 21st, two Travis builders' cache for Docker are suddenly invalidated. Normally this
# is fine, because we just need to rebuild the Docker image. However, it reveals a network issue:
# downloading from `ftp://gcc.gnu.org/` from Travis (using passive mode) often leads to "Connection
# timed out" error, and even when the download completed, the file is usually corrupted. This causes
# nothing to be landed that day.
#
# We observed that the `gcc-4.8.5.tar.bz2` above can be downloaded successfully, so as a stability
# improvement we try to download from the HTTPS mirror instead. Turns out this uncovered the third
# bug: the host `gcc.gnu.org` and `cygwin.com` share the same IP, and the TLS certificate of the
# latter host is presented to `wget`! Therefore, we choose to download from the insecure HTTP server
# instead here.
#
sed -i'' 's|ftp://gcc\.gnu\.org/|http://gcc.gnu.org/|g' ./contrib/download_prerequisites

./contrib/download_prerequisites
mkdir ../gcc-build
cd ../gcc-build
hide_output ../gcc-$GCC/configure \
    --prefix=/rustroot \
    --enable-languages=c,c++ \
    --disable-gnu-unique-object
hide_output make -j10
hide_output make install
ln -s gcc /rustroot/bin/cc

cd ..
rm -rf gcc-build
rm -rf gcc-$GCC
