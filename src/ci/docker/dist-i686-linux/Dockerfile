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
COPY dist-i686-linux/shared.sh dist-i686-linux/build-binutils.sh /tmp/

# We need a build of openssl which supports SNI to download artifacts from
# static.rust-lang.org. This'll be used to link into libcurl below (and used
# later as well), so build a copy of OpenSSL with dynamic libraries into our
# generic root.
COPY dist-i686-linux/build-openssl.sh /tmp/
RUN ./build-openssl.sh

# The `curl` binary on CentOS doesn't support SNI which is needed for fetching
# some https urls we have, so install a new version of libcurl + curl which is
# using the openssl we just built previously.
#
# Note that we also disable a bunch of optional features of curl that we don't
# really need.
COPY dist-i686-linux/build-curl.sh /tmp/
RUN ./build-curl.sh

# binutils < 2.22 has a bug where the 32-bit executables it generates
# immediately segfault in Rust, so we need to install our own binutils.
#
# See https://github.com/rust-lang/rust/issues/20440 for more info
RUN ./build-binutils.sh

# Need a newer version of gcc than centos has to compile LLVM nowadays
COPY dist-i686-linux/build-gcc.sh /tmp/
RUN ./build-gcc.sh

# CentOS 5.5 has Python 2.4 by default, but LLVM needs 2.7+
COPY dist-i686-linux/build-python.sh /tmp/
RUN ./build-python.sh

# Apparently CentOS 5.5 desn't have `git` in yum, but we're gonna need it for
# cloning, so download and build it here.
COPY dist-i686-linux/build-git.sh /tmp/
RUN ./build-git.sh

# libssh2 (a dependency of Cargo) requires cmake 2.8.11 or higher but CentOS
# only has 2.6.4, so build our own
COPY dist-i686-linux/build-cmake.sh /tmp/
RUN ./build-cmake.sh

# for sanitizers, we need kernel headers files newer than the ones CentOS ships
# with so we install newer ones here
COPY dist-i686-linux/build-headers.sh /tmp/
RUN ./build-headers.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV HOSTS=i686-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS \
      --host=$HOSTS \
      --enable-extended \
      --enable-sanitizers \
      --enable-profiler
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS

# This is the only builder which will create source tarballs
ENV DIST_SRC 1

# When we build cargo in this container, we don't want it to use the system
# libcurl, instead it should compile its own.
ENV LIBCURL_NO_PKG_CONFIG 1
