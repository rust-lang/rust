FROM ubuntu:16.04

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
  xz-utils \
  libssl-dev \
  pkg-config

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu --set rust.ignore-git=false
ENV SCRIPT python3 ../x.py --stage 2 test distcheck
ENV DIST_SRC 1
