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
  libzstd-dev \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS \
 --build=x86_64-unknown-linux-gnu \
 --enable-sanitizers \
 --enable-profiler \
 --enable-compiler-docs \
 --set llvm.libzstd=true

# Build the toolchain with multiple parallel frontend threads and then run tests
# Tests are still compiled serially at the moment (intended to be changed in follow-ups).
ENV SCRIPT python3 ../x.py --stage 2 test --set rust.parallel-frontend-threads=4
