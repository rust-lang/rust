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
sparcv9)
        SYSROOT_MACH='sparc'
        ;;
*)
        printf 'ERROR: unknown architecture: %s\n' "$ARCH"
        exit 1
esac

BUILD_TARGET="$ARCH-pc-solaris2.11"

#
# The illumos and the Solaris build both use the same GCC-level host triple,
# though different versions of GCC are used and with different configuration
# options.  To ensure as little accidental cross-pollination as possible, we
# build the illumos toolchain in a specific directory tree and just symlink the
# expected tools into /usr/local/bin at the end.  We omit /usr/local/bin from
# PATH here for similar reasons.
#
PREFIX="/opt/solaris/$ARCH"
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
GCC_URL="https://ci-mirrors.rust-lang.org/rustc/$GCC_TAR"

SYSROOT_VER='2025-02-21'
if [ $ARCH = "x86_64" ]; then
SYSROOT_SUM='e82b78c14464cc2dc71f3cdab312df3dd63441d7c23eeeaf34d41d8b947688d3'
SYSROOT_TAR="solaris-11.4.42.111.0-i386-sysroot-v$SYSROOT_VER.tar.bz2"
SYSROOT_DIR="$PREFIX/sysroot-x86_64"
else
SYSROOT_SUM='e249a7ef781b9b3297419bd014fa0574800703981d84e113d6af3a897a8b4ffc'
SYSROOT_TAR="solaris-11.4.42.111.0-sparc-sysroot-v$SYSROOT_VER.tar.bz2"
SYSROOT_DIR="$PREFIX/sysroot-sparcv9"
fi
SYSROOT_URL="https://ci-mirrors.rust-lang.org/rustc/$SYSROOT_TAR"

BINUTILS_VERSION='2.44'
BINUTILS_SUM='ce2017e059d63e67ddb9240e9d4ec49c2893605035cd60e92ad53177f4377237'
BINUTILS_BASE="binutils-$BINUTILS_VERSION"
BINUTILS_TAR="$BINUTILS_BASE.tar.xz"
BINUTILS_URL="https://ci-mirrors.rust-lang.org/rustc/$BINUTILS_TAR"


case "$PHASE" in
sysroot)
        download_tar_and_extract_into_dir "$SYSROOT_URL" "$SYSROOT_SUM" "$SYSROOT_DIR"
        ;;

binutils)
        download_tar_and_extract_into_dir "$BINUTILS_URL" "$BINUTILS_SUM" /ws/src/binutils
        cat > binutils.patch <<EOF
Workaround for: https://github.com/rust-lang/rust/issues/137997
--- binutils-2.44/bfd/elflink.c
+++ binutils-2.44/bfd/elflink.c
@@ -5150,7 +5150,7 @@
          if it is not a function, because it might be the version
          symbol itself.  FIXME: What if it isn't?  */
       if ((iver.vs_vers & VERSYM_HIDDEN) != 0
-          || (vernum > 1
+          || (vernum > 1 && strcmp(name, "logb") != 0
           && (!bfd_is_abs_section (sec)
               || bed->is_function_type (ELF_ST_TYPE (isym->st_info)))))
         {
EOF
        f=binutils-$BINUTILS_VERSION/bfd/elflink.c && expand -t 4 "$f" > "$f.exp"
        mv binutils-$BINUTILS_VERSION/bfd/elflink.c.exp binutils-$BINUTILS_VERSION/bfd/elflink.c
        patch binutils-$BINUTILS_VERSION/bfd/elflink.c < binutils.patch
        rm binutils.patch

        mkdir -p /ws/build/binutils
        cd /ws/build/binutils
        "/ws/src/binutils/$BINUTILS_BASE/configure" \
            --prefix="$PREFIX" \
            --target="$BUILD_TARGET" \
            --program-prefix="$ARCH-solaris-" \
            --with-sysroot="$SYSROOT_DIR"

        make -j "$JOBS"

        mkdir -p "$PREFIX"
        make install

        cd
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
            --program-prefix="$ARCH-solaris-" \
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
        (cd "$PREFIX/bin" && ls -U) | grep "^$ARCH-solaris-" |
            xargs -t -I% ln -s "$PREFIX/bin/%" '/usr/local/bin/'

        cd
        rm -rf /ws/src/gcc /ws/build/gcc
        ;;

*)
        printf 'ERROR: unknown phase "%s"\n' "$PHASE" >&2
        exit 100
        ;;
esac
