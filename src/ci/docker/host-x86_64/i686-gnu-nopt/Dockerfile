FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-multilib \
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
  zlib1g-dev \
  lib32z1-dev \
  xz-utils


COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

RUN mkdir -p /config
RUN echo "[rust]" > /config/nopt-std-config.toml
RUN echo "optimize = false" >> /config/nopt-std-config.toml

# We are intentionally allowing an old toolchain on this builder (and that's
# incompatible with LLVM downloads today).
ENV NO_DOWNLOAD_CI_LLVM 1

ENV RUST_CONFIGURE_ARGS --build=i686-unknown-linux-gnu --disable-optimize-tests \
    --set llvm.allow-old-toolchain
ENV SCRIPT python3 ../x.py test --stage 0 --config /config/nopt-std-config.toml library/std \
  && python3 ../x.py --stage 2 test
