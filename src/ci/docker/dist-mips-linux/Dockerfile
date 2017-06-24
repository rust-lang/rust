FROM rust-ci-base

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-mips-linux-gnu

ENV HOSTS=mips-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
