FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-multilib \
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
  zlib1g-dev \
  lib32z1-dev \
  xz-utils \
  mingw-w64 \
  && rm -rf /var/lib/apt/lists/*


COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS --build=i686-unknown-linux-gnu
COPY scripts/stage_2_test_set1.sh /scripts/
COPY scripts/stage_2_test_set2.sh /scripts/
ENV SCRIPT "Must specify DOCKER_SCRIPT for this image"
