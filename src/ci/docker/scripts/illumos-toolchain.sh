#!/bin/bash

set -o errexit
set -o pipefail
set -o xtrace

source /tmp/shared.sh

ARCH="$1"
PHASE="$2"

JOBS="$(getconf _NPROCESSORS_ONLN)"

case "$ARCH" in
x86_64)
        SYSROOT_MACH='i386'
        ;;
*)
        printf 'ERROR: unknown architecture: %s\n' "$ARCH"
        exit 1
esac

BUILD_TARGET="$ARCH-pc-solaris2.10"

#
# The illumos and the Solaris build both use the same GCC-level host triple,
# though different versions of GCC are used and with different configure
# options.  To ensure as little accidental cross-pollination as possible, we
# build the illumos toolchain in a specific directory tree and just symlink the
# expected tools into /usr/local/bin at the end.  We omit /usr/local/bin from
# PATH here for similar reasons.
#
PREFIX="/opt/illumos/$ARCH"
export PATH="$PREFIX/bin:/usr/bin:/bin:/usr/sbin:/sbin"

#
# NOTE: The compiler version selected here is more specific than might appear.
# GCC 7.X releases do not appear to cross-compile correctly for Solaris
# targets, at least insofar as they refuse to enable TLS in libstdc++.  When
# changing the GCC version in future, one must carefully verify that TLS is
# enabled in all of the static libraries we intend to include in output
# binaries.
#
GCC_VERSION='8.4.0'
GCC_SUM='e30a6e52d10e1f27ed55104ad233c30bd1e99cfb5ff98ab022dc941edd1b2dd4'
GCC_BASE="gcc-$GCC_VERSION"
GCC_TAR="gcc-$GCC_VERSION.tar.xz"
GCC_URL="https://ftp.gnu.org/gnu/gcc/$GCC_BASE/$GCC_TAR"

SYSROOT_VER='20181213-de6af22ae73b-v1'
SYSROOT_SUM='ee792d956dfa6967453cebe9286a149143290d296a8ce4b8a91d36bea89f8112'
SYSROOT_TAR="illumos-sysroot-$SYSROOT_MACH-$SYSROOT_VER.tar.gz"
SYSROOT_URL='https://github.com/illumos/sysroot/releases/download/'
SYSROOT_URL+="$SYSROOT_VER/$SYSROOT_TAR"
SYSROOT_DIR="$PREFIX/sysroot"

BINUTILS_VERSION='2.40'
BINUTILS_SUM='f8298eb153a4b37d112e945aa5cb2850040bcf26a3ea65b5a715c83afe05e48a'
BINUTILS_BASE="binutils-$BINUTILS_VERSION"
BINUTILS_TAR="$BINUTILS_BASE.tar.bz2"
BINUTILS_URL="https://ftp.gnu.org/gnu/binutils/$BINUTILS_TAR"


case "$PHASE" in
sysroot)
        download_tar_and_extract_into_dir "$SYSROOT_URL" "$SYSROOT_SUM" "$SYSROOT_DIR"
        ;;

binutils)
        download_tar_and_extract_into_dir "$BINUTILS_URL" "$BINUTILS_SUM" /ws/src/binutils
        mkdir -p /ws/build/binutils
        cd /ws/build/binutils
        "/ws/src/binutils/$BINUTILS_BASE/configure" \
            --prefix="$PREFIX" \
            --target="$BUILD_TARGET" \
            --program-prefix="$ARCH-illumos-" \
            --with-sysroot="$SYSROOT_DIR"

        make -j "$JOBS"

        mkdir -p "$PREFIX"
        make install

        cd /
        rm -rf /ws/src/binutils /ws/build/binutils
        ;;

gcc)
        download_tar_and_extract_into_dir "$GCC_URL" "$GCC_SUM" /ws/src/gcc
        mkdir -p /ws/build/gcc
        cd /ws/build/gcc
        export CFLAGS='-fPIC'
        export CXXFLAGS='-fPIC'
        export CXXFLAGS_FOR_TARGET='-fPIC'
        export CFLAGS_FOR_TARGET='-fPIC'
        "/ws/src/gcc/$GCC_BASE/configure" \
            --prefix="$PREFIX" \
            --target="$BUILD_TARGET" \
            --program-prefix="$ARCH-illumos-" \
            --with-sysroot="$SYSROOT_DIR" \
            --with-gnu-as \
            --with-gnu-ld \
            --disable-nls \
            --disable-libgomp \
            --disable-libquadmath \
            --disable-libssp \
            --disable-libvtv \
            --disable-libcilkrts \
            --disable-libada \
            --disable-libsanitizer \
            --disable-libquadmath-support \
            --disable-shared \
            --enable-tls

        make -j "$JOBS"

        mkdir -p "$PREFIX"
        make install

        #
        # Link toolchain commands into /usr/local/bin so that cmake and others
        # can find them:
        #
        (cd "$PREFIX/bin" && ls -U) | grep "^$ARCH-illumos-" |
            xargs -t -I% ln -s "$PREFIX/bin/%" '/usr/local/bin/'

        cd /
        rm -rf /ws/src/gcc /ws/build/gcc
        ;;

*)
        printf 'ERROR: unknown phase "%s"\n' "$PHASE" >&2
        exit 100
        ;;
esac
