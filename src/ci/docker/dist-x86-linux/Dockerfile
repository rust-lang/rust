FROM centos:5

WORKDIR /build

RUN yum upgrade -y && yum install -y \
      curl \
      bzip2 \
      gcc \
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
COPY shared.sh build-binutils.sh /tmp/

# We need a build of openssl which supports SNI to download artifacts from
# static.rust-lang.org. This'll be used to link into libcurl below (and used
# later as well), so build a copy of OpenSSL with dynamic libraries into our
# generic root.
COPY build-openssl.sh /tmp/
RUN ./build-openssl.sh

# The `curl` binary on CentOS doesn't support SNI which is needed for fetching
# some https urls we have, so install a new version of libcurl + curl which is
# using the openssl we just built previously.
#
# Note that we also disable a bunch of optional features of curl that we don't
# really need.
COPY build-curl.sh /tmp/
RUN ./build-curl.sh

# binutils < 2.22 has a bug where the 32-bit executables it generates
# immediately segfault in Rust, so we need to install our own binutils.
#
# See https://github.com/rust-lang/rust/issues/20440 for more info
RUN ./build-binutils.sh

# Need a newer version of gcc than centos has to compile LLVM nowadays
COPY build-gcc.sh /tmp/
RUN ./build-gcc.sh

# CentOS 5.5 has Python 2.4 by default, but LLVM needs 2.7+
COPY build-python.sh /tmp/
RUN ./build-python.sh

# Apparently CentOS 5.5 desn't have `git` in yum, but we're gonna need it for
# cloning, so download and build it here.
COPY build-git.sh /tmp/
RUN ./build-git.sh

# libssh2 (a dependency of Cargo) requires cmake 2.8.11 or higher but CentOS
# only has 2.6.4, so build our own
COPY build-cmake.sh /tmp/
RUN ./build-cmake.sh

# for sanitizers, we need kernel headers files newer than the ones CentOS ships
# with so we install newer ones here
COPY build-headers.sh /tmp/
RUN ./build-headers.sh

RUN curl -Lo /rustroot/dumb-init \
      https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64 && \
      chmod +x /rustroot/dumb-init
ENTRYPOINT ["/rustroot/dumb-init", "--"]

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-02-25-sccache-x86_64-unknown-linux-musl && \
      chmod +x /usr/local/bin/sccache

ENV HOSTS=i686-unknown-linux-gnu
ENV HOSTS=$HOSTS,x86_64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS \
      --host=$HOSTS \
      --enable-extended \
      --enable-sanitizers
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
ENV DIST_SRC 1
