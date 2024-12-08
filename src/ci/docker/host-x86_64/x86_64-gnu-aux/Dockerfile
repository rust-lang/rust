FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  libssl-dev \
  sudo \
  xz-utils \
  pkg-config \
  libgl1-mesa-dev \
  llvm-dev \
  libfreetype6-dev \
  libexpat1-dev \
  tidy \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# Miri is just too slow with full assertions
ENV NO_DEBUG_ASSERTIONS=1
ENV NO_OVERFLOW_CHECKS=1

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu
ENV RUST_CHECK_TARGET check-aux
