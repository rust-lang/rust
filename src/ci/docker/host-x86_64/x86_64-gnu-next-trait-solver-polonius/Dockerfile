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
  sudo \
  gdb \
  libssl-dev \
  pkg-config \
  xz-utils \
  mingw-w64 \
  zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS="--build=x86_64-unknown-linux-gnu"

COPY scripts/x86_64-gnu-next-trait-solver-polonius.sh /scripts/
ENV SCRIPT="/scripts/x86_64-gnu-next-trait-solver-polonius.sh"
