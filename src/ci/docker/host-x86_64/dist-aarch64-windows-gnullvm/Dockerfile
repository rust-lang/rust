FROM ubuntu:24.04

WORKDIR /build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    g++ \
    git \
    make \
    ninja-build \
    python3 \
    xz-utils

ENV ARCH=aarch64
COPY host-x86_64/dist-x86_64-windows-gnullvm/install-llvm-mingw.sh /build
RUN ./install-llvm-mingw.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV CC_aarch64_pc_windows_gnullvm=aarch64-w64-mingw32-clang \
    CXX_aarch64_pc_windows_gnullvm=aarch64-w64-mingw32-clang++

ENV HOST=aarch64-pc-windows-gnullvm

ENV RUST_CONFIGURE_ARGS \
    --enable-full-tools \
    --enable-profiler \
    --enable-sanitizers \
    --disable-docs

ENV SCRIPT python3 ../x.py dist --host $HOST --target $HOST
