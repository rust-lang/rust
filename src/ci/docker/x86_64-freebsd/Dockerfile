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
  ccache \
  sudo \
  bzip2 \
  xz-utils \
  wget

COPY build-toolchain.sh /tmp/
RUN sh /tmp/build-toolchain.sh

ENV \
    AR_x86_64_unknown_freebsd=x86_64-unknown-freebsd10-ar \
    CC_x86_64_unknown_freebsd=x86_64-unknown-freebsd10-gcc

ENV RUST_CONFIGURE_ARGS --target=x86_64-unknown-freebsd
ENV RUST_CHECK_TARGET ""
RUN mkdir /tmp/obj
RUN chmod 777 /tmp/obj
