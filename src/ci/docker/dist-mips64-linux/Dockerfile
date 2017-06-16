FROM rust-ci-base

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-mips64-linux-gnuabi64

ENV HOSTS=mips64-unknown-linux-gnuabi64

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
