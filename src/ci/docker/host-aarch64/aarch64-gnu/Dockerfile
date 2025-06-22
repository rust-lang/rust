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
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS \
 --build=aarch64-unknown-linux-gnu \
 --enable-sanitizers \
 --enable-profiler \
 --enable-compiler-docs
ENV SCRIPT python3 ../x.py --stage 2 test && \
  python3 ../x.py --stage 2 test src/tools/cargo
