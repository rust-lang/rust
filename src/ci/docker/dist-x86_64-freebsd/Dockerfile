FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  clang \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  sudo \
  bzip2 \
  xz-utils \
  wget \
  libssl-dev \
  pkg-config

COPY scripts/freebsd-toolchain.sh /tmp/
RUN /tmp/freebsd-toolchain.sh x86_64

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_x86_64_unknown_freebsd=x86_64-unknown-freebsd10-ar \
    CC_x86_64_unknown_freebsd=x86_64-unknown-freebsd10-clang \
    CXX_x86_64_unknown_freebsd=x86_64-unknown-freebsd10-clang++

ENV HOSTS=x86_64-unknown-freebsd

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
