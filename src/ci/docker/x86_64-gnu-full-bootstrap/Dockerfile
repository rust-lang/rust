FROM rust-ci-base

ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --enable-full-bootstrap
ENV SCRIPT python2.7 ../x.py build
