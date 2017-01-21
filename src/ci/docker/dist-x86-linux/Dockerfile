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
      pkg-config \
      wget \
      autoconf \
      gettext

ENV PATH=/rustroot/bin:$PATH
ENV LD_LIBRARY_PATH=/rustroot/lib64:/rustroot/lib
WORKDIR /tmp

# binutils < 2.22 has a bug where the 32-bit executables it generates
# immediately segfault in Rust, so we need to install our own binutils.
#
# See https://github.com/rust-lang/rust/issues/20440 for more info
COPY shared.sh build-binutils.sh /tmp/
RUN ./build-binutils.sh

# Need a newer version of gcc than centos has to compile LLVM nowadays
COPY build-gcc.sh /tmp/
RUN ./build-gcc.sh

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

RUN curl -Lo /rustroot/dumb-init \
      https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64 && \
      chmod +x /rustroot/dumb-init
ENTRYPOINT ["/rustroot/dumb-init", "--"]

ENV SCCACHE_DIGEST=7237e38e029342fa27b7ac25412cb9d52554008b12389727320bd533fd7f05b6a96d55485f305caf95e5c8f5f97c3313e10012ccad3e752aba2518f3522ba783
RUN curl -L https://api.pub.build.mozilla.org/tooltool/sha512/$SCCACHE_DIGEST | \
      xz --decompress | \
      tar xf - -C /usr/local/bin --strip-components=1

ENV HOSTS=i686-unknown-linux-gnu
ENV HOSTS=$HOSTS,x86_64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --host=$HOSTS
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
