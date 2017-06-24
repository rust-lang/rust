FROM rust-ci-base

RUN apt-get update && apt-get install -y --no-install-recommends \
  llvm-3.7-tools

ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --llvm-root=/usr/lib/llvm-3.7
ENV RUST_CHECK_TARGET check
