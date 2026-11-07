# We use the ghcr base image because ghcr doesn't have a rate limit
# and the tidy job doesn't cache docker images in CI.
FROM ghcr.io/rust-lang/ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  python3-venv \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  libssl-dev \
  pkg-config \
  mingw-w64 \
  && rm -rf /var/lib/apt/lists/*

ENV PUPPETEER_SKIP_DOWNLOAD 1

COPY scripts/nodejs.sh /scripts/
RUN sh /scripts/nodejs.sh /node
ENV PATH="/node/bin:${PATH}"

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY host-x86_64/test-pr-check-1/validate-toolstate.sh /scripts/

# python3 resolves to python3.12 on Ubuntu 24.04
ENV SCRIPT="python3 ../x.py test src/tools/tidy tidyselftest --extra-checks=py,cpp,js,spellcheck"
