FROM centos:5

WORKDIR /build

# Centos 5 is EOL and is no longer available from the usual mirrors, so switch
# to http://vault.centos.org/
RUN sed -i 's/enabled=1/enabled=0/' /etc/yum/pluginconf.d/fastestmirror.conf
RUN sed -i 's/mirrorlist/#mirrorlist/' /etc/yum.repos.d/*.repo
RUN sed -i 's|#\(baseurl.*\)mirror.centos.org/centos/$releasever|\1vault.centos.org/5.11|' /etc/yum.repos.d/*.repo

RUN yum upgrade -y && yum install -y \
      curl \
      bzip2 \
      gcc \
      gcc-c++ \
      make \
      glibc-devel \
      perl \
      zlib-devel \
      file \
      xz \
      which \
      pkgconfig \
      wget \
      autoconf \
      gettext

ENV PATH=/rustroot/bin:$PATH
ENV LD_LIBRARY_PATH=/rustroot/lib64:/rustroot/lib
ENV PKG_CONFIG_PATH=/rustroot/lib/pkgconfig
WORKDIR /tmp
COPY dist-x86_64-linux/shared.sh /tmp/

# We need a build of openssl which supports SNI to download artifacts from
# static.rust-lang.org. This'll be used to link into libcurl below (and used
# later as well), so build a copy of OpenSSL with dynamic libraries into our
# generic root.
COPY dist-x86_64-linux/build-openssl.sh /tmp/
RUN ./build-openssl.sh

# The `curl` binary on CentOS doesn't support SNI which is needed for fetching
# some https urls we have, so install a new version of libcurl + curl which is
# using the openssl we just built previously.
#
# Note that we also disable a bunch of optional features of curl that we don't
# really need.
COPY dist-x86_64-linux/build-curl.sh /tmp/
RUN ./build-curl.sh

# binutils < 2.22 has a bug where the 32-bit executables it generates
# immediately segfault in Rust, so we need to install our own binutils.
#
# See https://github.com/rust-lang/rust/issues/20440 for more info
COPY dist-x86_64-linux/build-binutils.sh /tmp/
RUN ./build-binutils.sh

# libssh2 (a dependency of Cargo) requires cmake 2.8.11 or higher but CentOS
# only has 2.6.4, so build our own
COPY dist-x86_64-linux/build-cmake.sh /tmp/
RUN ./build-cmake.sh

# Build a version of gcc capable of building LLVM 6
COPY dist-x86_64-linux/build-gcc.sh /tmp/
RUN ./build-gcc.sh

# CentOS 5.5 has Python 2.4 by default, but LLVM needs 2.7+
COPY dist-x86_64-linux/build-python.sh /tmp/
RUN ./build-python.sh

# Now build LLVM+Clang 7, afterwards configuring further compilations to use the
# clang/clang++ compilers.
COPY dist-x86_64-linux/build-clang.sh /tmp/
RUN ./build-clang.sh
ENV CC=clang CXX=clang++

# Apparently CentOS 5.5 desn't have `git` in yum, but we're gonna need it for
# cloning, so download and build it here.
COPY dist-x86_64-linux/build-git.sh /tmp/
RUN ./build-git.sh

# for sanitizers, we need kernel headers files newer than the ones CentOS ships
# with so we install newer ones here
COPY dist-x86_64-linux/build-headers.sh /tmp/
RUN ./build-headers.sh

# OpenSSL requires a more recent version of perl
# with so we install newer ones here
COPY dist-x86_64-linux/build-perl.sh /tmp/
RUN ./build-perl.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV HOSTS=x86_64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS \
      --enable-full-tools \
      --enable-sanitizers \
      --enable-profiler \
      --enable-compiler-docs \
      --set target.x86_64-unknown-linux-gnu.linker=clang \
      --set target.x86_64-unknown-linux-gnu.ar=/rustroot/bin/llvm-ar \
      --set target.x86_64-unknown-linux-gnu.ranlib=/rustroot/bin/llvm-ranlib \
      --set llvm.thin-lto=true \
      --set rust.jemalloc
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=clang

# This is the only builder which will create source tarballs
ENV DIST_SRC 1

# When we build cargo in this container, we don't want it to use the system
# libcurl, instead it should compile its own.
ENV LIBCURL_NO_PKG_CONFIG 1

ENV DIST_REQUIRE_ALL_TOOLS 1
