FROM ubuntu:26.04

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
  clang \
  && rm -rf /var/lib/apt/lists/*

# Install Intel SDE for AVX-512 emulation
RUN curl -L http://ci-mirrors.rust-lang.org/sde-external-10.8.0-2026-03-15-lin.tar.xz -o /tmp/sde.tar.xz \
  && mkdir -p /intel-sde \
  && tar -xJf /tmp/sde.tar.xz --strip-components=1 -C /intel-sde \
  && rm /tmp/sde.tar.xz

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS="--build=x86_64-unknown-linux-gnu \
 --enable-sanitizers \
 --enable-profiler \
 --enable-compiler-docs \
 --set llvm.libzstd=true"
ENV SCRIPT="python3 ../x.py --stage 2 test"
