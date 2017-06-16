FROM rust-ci-base

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-multilib

ENV RUST_CONFIGURE_ARGS --build=i686-unknown-linux-gnu
ENV SCRIPT python2.7 ../x.py test
