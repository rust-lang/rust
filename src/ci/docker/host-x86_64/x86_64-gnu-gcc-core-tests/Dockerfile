# This Dockerfile builds sysroot crates with the GCC backend and then runs
# libcore tests with the GCC backend.

FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  bzip2 \
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
  # libgccjit dependencies
  flex \
  libmpfr-dev \
  libgmp-dev \
  libmpc3 \
  libmpc-dev \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV CODEGEN_BACKENDS="gcc"
ENV NO_DEBUG_ASSERTIONS="1"
ENV RUSTFLAGS="-Cpanic=abort -Zpanic-abort-tests"
ENV RUST_CONFIGURE_ARGS="--build=x86_64-unknown-linux-gnu \
 --enable-sanitizers \
 --enable-profiler \
 --enable-compiler-docs \
 --set llvm.libzstd=true"
ENV SCRIPT="python3 ../x.py \
  --stage 1 \
  test library/coretests"
