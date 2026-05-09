FROM ubuntu:25.10

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
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
  xz-utils \
  clang \
  lld \
  && rm -rf /var/lib/apt/lists/* \
  && find /usr/lib -maxdepth 5 -name 'clang++' -executable 2>/dev/null \
     | sort -V | tail -1 | xargs -r ln -sf -t /usr/local/bin/ \
  && if ! test -x /usr/local/bin/clang++; then \
       find /usr/bin -name 'clang++-*' -executable 2>/dev/null \
       | sort -V | tail -1 | xargs -r -I{} ln -sf {} /usr/local/bin/clang++; \
     fi \
  && clang++ --version
  
COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS="--build=aarch64-unknown-linux-gnu \
 --enable-sanitizers \
 --enable-profiler \
 --enable-compiler-docs"
ENV SCRIPT="python3 ../x.py --stage 2 test && \
  python3 ../x.py --stage 2 test src/tools/cargo"
