FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# libclang1 is required for libclang.so, required by bindgen
# clang, llvm and lld is required by RfL to compile host code
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
  flex \
  bison \
  bc \
  clang-15 \
  libclang1-15 \
  llvm-15 \
  lld-15 \
  libelf-dev \
  libedit-dev \
  libssl-dev \
  pkg-config \
  zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# RfL needs access to cland, lld and llvm tools
ENV PATH="${PATH}:/usr/lib/llvm-15/bin"

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu

COPY /scripts/rfl-build.sh /tmp/rfl-build.sh
ENV SCRIPT bash /tmp/rfl-build.sh
