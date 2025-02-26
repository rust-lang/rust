#!/usr/bin/env bash
set -eux

source shared.sh

# Note: in the future when bumping to version 10.1.0, also take care of the sed block below.
# This version is specified in the Dockerfile
GCC=$GCC_VERSION

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
# Note: in version 10.1.0, the URL used in `download_prerequisites` has changed from using FTP to
# using HTTP. When bumping to that gcc version, we can likely remove the sed replacement below, or
# the expression will need to be updated. That new URL is available at:
# https://github.com/gcc-mirror/gcc/blob/6e6e3f144a33ae504149dc992453b4f6dea12fdb/contrib/download_prerequisites#L35
#
sed -i'' 's|ftp://gcc\.gnu\.org/|https://gcc.gnu.org/|g' ./contrib/download_prerequisites

./contrib/download_prerequisites
mkdir ../gcc-build
cd ../gcc-build

# '-fno-reorder-blocks-and-partition' is required to
# enable BOLT optimization of the C++ standard library,
# which is included in librustc_driver.so
hide_output ../gcc-$GCC/configure \
    --prefix=/rustroot \
    --enable-languages=c,c++ \
    --disable-gnu-unique-object \
    --enable-cxx-flags='-fno-reorder-blocks-and-partition'
hide_output make -j$(nproc)
hide_output make install
ln -s gcc /rustroot/bin/cc

cd ..
rm -rf gcc-build
rm -rf gcc-$GCC

if [[ $GCC_BUILD_TARGET == "i686-pc-linux-gnu" ]]; then
    # FIXME: clang doesn't find 32-bit libraries in /rustroot/lib,
    # but it does look all the way under /rustroot/lib/[...]/32,
    # so we can link stuff there to help it out.
    ln /rustroot/lib/*.{a,so} -rst /rustroot/lib/gcc/x86_64-pc-linux-gnu/$GCC/32/
fi
