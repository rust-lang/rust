FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  clang \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  sudo \
  bzip2 \
  xz-utils \
  wget \
  libssl-dev \
  pkg-config

COPY scripts/freebsd-toolchain.sh /tmp/
RUN /tmp/freebsd-toolchain.sh i686

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_i686_unknown_freebsd=i686-unknown-freebsd10-ar \
    CC_i686_unknown_freebsd=i686-unknown-freebsd10-clang \
    CXX_i686_unknown_freebsd=i686-unknown-freebsd10-clang++

ENV HOSTS=i686-unknown-freebsd

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
