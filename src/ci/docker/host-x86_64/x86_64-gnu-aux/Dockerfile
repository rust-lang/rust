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
  libssl-dev \
  sudo \
  xz-utils \
  pkg-config \
  libgl1-mesa-dev \
  llvm-dev \
  libfreetype6-dev \
  libexpat1-dev \
  tidy

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

# We are intentionally allowing an old toolchain on this builder (and that's
# incompatible with LLVM downloads today).
ENV NO_DOWNLOAD_CI_LLVM 1

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu \
    --set llvm.allow-old-toolchain
ENV RUST_CHECK_TARGET check-aux
