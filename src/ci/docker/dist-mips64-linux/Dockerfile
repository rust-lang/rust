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
  gdb \
  xz-utils \
  g++-mips64-linux-gnuabi64 \
  libssl-dev \
  pkg-config

COPY scripts/dumb-init.sh /scripts/
RUN sh /scripts/dumb-init.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENTRYPOINT ["/usr/bin/dumb-init", "--"]

ENV HOSTS=mips64-unknown-linux-gnuabi64

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
