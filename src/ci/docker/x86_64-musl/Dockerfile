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
  xz-utils \
  sudo \
  gdb

WORKDIR /build/
COPY build-musl.sh /build/
RUN sh /build/build-musl.sh && rm -rf /build

ENV RUST_CONFIGURE_ARGS \
      --target=x86_64-unknown-linux-musl \
      --musl-root=/musl-x86_64
ENV RUST_CHECK_TARGET check-stage2-T-x86_64-unknown-linux-musl-H-x86_64-unknown-linux-gnu

RUN mkdir /tmp/obj
RUN chmod 777 /tmp/obj
