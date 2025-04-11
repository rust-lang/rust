#!/bin/bash

set -o errexit
set -o pipefail
set -o xtrace

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
# though different versions of GCC are used and with different configure
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
GCC_URL="https://ftp.gnu.org/gnu/gcc/$GCC_BASE/$GCC_TAR"

if [ $ARCH = "x86_64" ]; then
SYSROOT_VER='2025-02-21'
SYSROOT_SUM='e82b78c14464cc2dc71f3cdab312df3dd63441d7c23eeeaf34d41d8b947688d3'
SYSROOT_TAR="solaris-11.4.42.111.0-i386-sysroot-v$SYSROOT_VER.tar.bz2"
SYSROOT_URL="https://github.com/psumbera/solaris-sysroot/releases/download/v$SYSROOT_VER/"
SYSROOT_URL+="$SYSROOT_TAR"
SYSROOT_DIR="$PREFIX/sysroot-x86_64"
else
SYSROOT_VER='2025-02-21'
SYSROOT_SUM='e249a7ef781b9b3297419bd014fa0574800703981d84e113d6af3a897a8b4ffc'
SYSROOT_TAR="solaris-11.4.42.111.0-sparc-sysroot-v$SYSROOT_VER.tar.bz2"
SYSROOT_URL="https://github.com/psumbera/solaris-sysroot/releases/download/v$SYSROOT_VER/"
SYSROOT_URL+="$SYSROOT_TAR"
SYSROOT_DIR="$PREFIX/sysroot-sparcv9"
fi

BINUTILS_VERSION='2.44'
BINUTILS_SUM='f66390a661faa117d00fab2e79cf2dc9d097b42cc296bf3f8677d1e7b452dc3a'
BINUTILS_BASE="binutils-$BINUTILS_VERSION"
BINUTILS_TAR="$BINUTILS_BASE.tar.bz2"
BINUTILS_URL="https://ftp.gnu.org/gnu/binutils/$BINUTILS_TAR"


download_file() {
        local file="$1"
        local url="$2"
        local sum="$3"

        while :; do
                if [[ -f "$file" ]]; then
                        if ! h="$(sha256sum "$file" | awk '{ print $1 }')"; then
                                printf 'ERROR: reading hash\n' >&2
                                exit 1
                        fi

                        if [[ "$h" == "$sum" ]]; then
                                return 0
                        fi

                        printf 'WARNING: hash mismatch: %s != expected %s\n' \
                            "$h" "$sum" >&2
                        rm -f "$file"
                fi

                printf 'Downloading: %s\n' "$url"
                if ! curl -f -L -o "$file" "$url"; then
                        rm -f "$file"
                        sleep 1
                fi
        done
}


case "$PHASE" in
sysroot)
        download_file "/tmp/$SYSROOT_TAR" "$SYSROOT_URL" "$SYSROOT_SUM"
        mkdir -p "$SYSROOT_DIR"
        cd "$SYSROOT_DIR"
        tar -xjf "/tmp/$SYSROOT_TAR"
        rm -f "/tmp/$SYSROOT_TAR"
        ;;

binutils)
        download_file "/tmp/$BINUTILS_TAR" "$BINUTILS_URL" "$BINUTILS_SUM"
        mkdir -p /ws/src/binutils
        cd /ws/src/binutils
        tar -xjf "/tmp/$BINUTILS_TAR"
        rm -f "/tmp/$BINUTILS_TAR"
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
        expand -t 4 binutils-2.44/bfd/elflink.c > binutils-2.44/bfd/elflink.c.exp
        mv binutils-2.44/bfd/elflink.c.exp binutils-2.44/bfd/elflink.c
        patch binutils-2.44/bfd/elflink.c < binutils.patch
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

        cd /
        rm -rf /ws/src/binutils /ws/build/binutils
        ;;

gcc)
        download_file "/tmp/$GCC_TAR" "$GCC_URL" "$GCC_SUM"
        mkdir -p /ws/src/gcc
        cd /ws/src/gcc
        tar -xJf "/tmp/$GCC_TAR"
        rm -f "/tmp/$GCC_TAR"

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

        cd /
        rm -rf /ws/src/gcc /ws/build/gcc
        ;;

*)
        printf 'ERROR: unknown phase "%s"\n' "$PHASE" >&2
        exit 100
        ;;
esac
