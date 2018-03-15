FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
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

COPY dist-i686-freebsd/build-toolchain.sh /tmp/
RUN /tmp/build-toolchain.sh i686

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_i686_unknown_freebsd=i686-unknown-freebsd10-ar \
    CC_i686_unknown_freebsd=i686-unknown-freebsd10-gcc \
    CXX_i686_unknown_freebsd=i686-unknown-freebsd10-g++

ENV HOSTS=i686-unknown-freebsd

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
