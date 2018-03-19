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
  libssl-dev \
  sudo \
  xz-utils \
  pkg-config

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY x86_64-gnu-tools/checkregression.py /tmp/
COPY x86_64-gnu-tools/checktools.sh /tmp/
COPY x86_64-gnu-tools/repo.sh /tmp/

ENV RUST_CONFIGURE_ARGS \
  --build=x86_64-unknown-linux-gnu \
  --enable-test-miri \
  --save-toolstates=/tmp/toolstates.json
ENV SCRIPT /tmp/checktools.sh ../x.py /tmp/toolstates.json linux
