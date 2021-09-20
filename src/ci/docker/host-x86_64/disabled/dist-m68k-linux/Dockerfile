FROM ubuntu:20.04

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
  gdb \
  xz-utils \
  g++-m68k-linux-gnu \
  libssl-dev \
  pkg-config


COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV HOSTS=m68k-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
