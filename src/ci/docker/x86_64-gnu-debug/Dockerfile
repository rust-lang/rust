FROM rust-ci-base

ENV RUST_CONFIGURE_ARGS \
      --build=x86_64-unknown-linux-gnu \
      --enable-debug \
      --enable-optimize
ENV SCRIPT python2.7 ../x.py build
