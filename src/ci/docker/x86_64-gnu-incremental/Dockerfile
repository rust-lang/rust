FROM rust-ci-base

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu
ENV RUSTFLAGS -Zincremental=/tmp/rust-incr-cache
ENV RUST_CHECK_TARGET check
