FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  python-minimal \
  git \
  cmake \
  ccache \
  libssl-dev \
  sudo

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu --enable-rustbuild
ENV RUST_CHECK_TARGET check-cargotest
RUN mkdir /tmp/obj
RUN chmod 777 /tmp/obj
