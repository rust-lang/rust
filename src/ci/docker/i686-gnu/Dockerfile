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
  sudo \
  gdb \
  xz-utils


COPY scripts/dumb-init.sh /scripts/
RUN sh /scripts/dumb-init.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENTRYPOINT ["/usr/bin/dumb-init", "--"]

ENV RUST_CONFIGURE_ARGS --build=i686-unknown-linux-gnu
ENV SCRIPT python2.7 ../x.py test
