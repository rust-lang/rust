FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-multilib \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  ccache \
  sudo \
  gdb

ENV RUST_CONFIGURE_ARGS --build=i686-unknown-linux-gnu --disable-optimize-tests
ENV RUST_CHECK_TARGET check
RUN mkdir /tmp/obj
RUN chmod 777 /tmp/obj
