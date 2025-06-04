FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  python3-pip \
  python3-pkg-resources \
  git \
  cmake \
  sudo \
  gdb \
  xz-utils \
  libssl-dev \
  pkg-config \
  mingw-w64 \
  && rm -rf /var/lib/apt/lists/*

ENV RUST_CONFIGURE_ARGS="--set rust.validate-mir-opts=3"

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV SCRIPT \
        python3 ../x.py check && \
        python3 ../x.py clippy ci && \
        python3 ../x.py test --stage 1 core alloc std test proc_macro && \
        # Build both public and internal documentation.
        RUSTDOCFLAGS=\"--document-private-items --document-hidden-items\" python3 ../x.py doc --stage 0 compiler && \
        RUSTDOCFLAGS=\"--document-private-items --document-hidden-items\" python3 ../x.py doc --stage 1 library && \
        mkdir -p /checkout/obj/staging/doc && \
        cp -r build/x86_64-unknown-linux-gnu/doc /checkout/obj/staging && \
        RUSTDOCFLAGS=\"--document-private-items --document-hidden-items\" python3 ../x.py doc --stage 1 library/test
