#!/bin/sh
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  automake \
  bison \
  bzip2 \
  ca-certificates \
  cmake \
  curl \
  file \
  flex \
  g++ \
  gawk \
  gdb \
  git \
  gperf \
  help2man \
  libncurses-dev \
  libssl-dev \
  libtool-bin \
  make \
  ninja-build \
  patch \
  pkg-config \
  python3 \
  rsync \
  sudo \
  texinfo \
  unzip \
  wget \
  xz-utils
