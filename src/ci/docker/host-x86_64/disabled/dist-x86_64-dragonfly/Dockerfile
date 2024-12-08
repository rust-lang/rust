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
  bzip2 \
  xz-utils \
  wget \
  libssl-dev \
  bsdtar \
  pkg-config


COPY host-x86_64/dist-x86_64-dragonfly/build-toolchain.sh /tmp/
COPY host-x86_64/dist-x86_64-dragonfly/patch-toolchain /tmp/
RUN /tmp/build-toolchain.sh /tmp/patch-toolchain

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_x86_64_unknown_dragonfly=x86_64-unknown-dragonfly-ar \
    CC_x86_64_unknown_dragonfly=x86_64-unknown-dragonfly-gcc \
    CXX_x86_64_unknown_dragonfly=x86_64-unknown-dragonfly-g++

ENV HOSTS=x86_64-unknown-dragonfly

ENV RUST_CONFIGURE_ARGS --enable-extended
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
