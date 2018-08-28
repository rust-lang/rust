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
  bzip2 \
  libssl-dev \
  pkg-config \
  gcc-arm-none-eabi


COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV TARGET=armv7r-none-eabihf

ENV RUST_CONFIGURE_ARGS --disable-docs

ENV SCRIPT python2.7 ../x.py dist --target $TARGET
