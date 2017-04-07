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
  xz-utils

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-04-04-sccache-x86_64-unknown-linux-musl && \
      chmod +x /usr/local/bin/sccache

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

ENV TARGETS=thumbv6m-none-eabi
ENV TARGETS=$TARGETS,thumbv7m-none-eabi
ENV TARGETS=$TARGETS,thumbv7em-none-eabi
ENV TARGETS=$TARGETS,thumbv7em-none-eabihf

ENV RUST_CONFIGURE_ARGS \
    --target=$TARGETS
ENV SCRIPT python2.7 ../x.py build --target $TARGETS src/libcore
