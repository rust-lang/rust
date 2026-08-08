FROM ubuntu:24.04

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
    pkg-config \
    xz-utils \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/x86_64-gnu-offload.sh /scripts/

ENV NO_DOWNLOAD_CI_LLVM 1
ENV CODEGEN_BACKENDS llvm


ENV RUST_CONFIGURE_ARGS \
    --build=x86_64-unknown-linux-gnu \
    --enable-clang \
    --enable-lld \
    --enable-llvm-offload \
    --enable-llvm-link-shared \
    --enable-ninja \
    --enable-option-checking \
    --disable-docs \
    --set llvm.download-ci-llvm=false

ENV SCRIPT /scripts/x86_64-gnu-offload.sh
