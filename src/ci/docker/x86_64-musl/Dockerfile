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

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

ENV RUST_CONFIGURE_ARGS \
      --target=x86_64-unknown-linux-musl \
      --musl-root-x86_64=/musl-x86_64
ENV RUST_CHECK_TARGET check-stage2-T-x86_64-unknown-linux-musl-H-x86_64-unknown-linux-gnu
ENV PATH=$PATH:/musl-x86_64/bin
ENV XPY_CHECK test --target x86_64-unknown-linux-musl

RUN mkdir /tmp/obj
RUN chmod 777 /tmp/obj
