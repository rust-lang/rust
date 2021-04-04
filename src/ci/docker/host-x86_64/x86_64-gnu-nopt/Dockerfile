FROM ubuntu:20.04

# Avoid interactive prompts while installing `tzdata` dependency with `DEBIAN_FRONTEND`.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  sudo \
  gdb \
  libssl-dev \
  pkg-config \
  xz-utils

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

RUN mkdir -p /config
RUN echo "[rust]" > /config/nopt-std-config.toml
RUN echo "optimize = false" >> /config/nopt-std-config.toml

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu \
  --disable-optimize-tests \
  --set rust.test-compare-mode
ENV SCRIPT python3 ../x.py test --stage 0 --config /config/nopt-std-config.toml library/std \
  && python3 ../x.py --stage 2 test
