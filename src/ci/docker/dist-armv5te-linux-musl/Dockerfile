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
  xz-utils \
  zlib1g-dev \
  g++-arm-linux-gnueabi \
  bzip2 \
  patch \
  pkg-config

WORKDIR /build

# Suppress some warnings in the openwrt toolchains we downloaded
ENV STAGING_DIR=/tmp

COPY scripts/musl.sh /build
RUN env \
    CC=arm-linux-gnueabi-gcc CFLAGS="-march=armv5te -marm -mfloat-abi=soft" \
    CXX=arm-linux-gnueabi-g++ CXXFLAGS="-march=armv5te -marm -mfloat-abi=soft" \
    bash musl.sh armv5te && \
    rm -rf /build/*

ENV TARGETS=armv5te-unknown-linux-musl

# FIXME: remove armv5te vars after https://github.com/alexcrichton/cc-rs/issues/271
#        get fixed and cc update
ENV CC_armv5te_unknown_linux_musl=arm-linux-gnueabi-gcc \
    CFLAGS_armv5te_unknown_linux_musl="-march=armv5te -marm -mfloat-abi=soft"

ENV RUST_CONFIGURE_ARGS \
      --musl-root-armv5te=/musl-armv5te \
      --disable-docs

ENV SCRIPT python2.7 ../x.py dist --target $TARGETS

# sccache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
