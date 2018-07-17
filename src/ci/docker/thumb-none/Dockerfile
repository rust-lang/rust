FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  sudo \
  xz-utils \
  zlib1g-dev \
  bzip2 \
  patch \
  libssl-dev \
  pkg-config \
  gcc-arm-none-eabi \
  libnewlib-arm-none-eabi

# ENV TARGETS=thumbv6m-none-eabi
ENV TARGETS=thumbv7m-none-eabi
ENV TARGETS=$TARGETS,thumbv7em-none-eabi
ENV TARGETS=$TARGETS,thumbv7em-none-eabihf

ENV RUST_CONFIGURE_ARGS \
      --disable-docs

ENV SCRIPT python2.7 ../x.py test --target $TARGETS src/test/run-make

# sccache
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh
