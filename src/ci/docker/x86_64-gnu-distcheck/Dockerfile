FROM rust-ci-base

ENV RUST_CONFIGURE_ARGS --build=x86_64-unknown-linux-gnu
ENV SCRIPT python2.7 ../x.py test distcheck
ENV DIST_SRC 1
