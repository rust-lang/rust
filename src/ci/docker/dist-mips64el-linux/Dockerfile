FROM rust-ci-base

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-mips64el-linux-gnuabi64

ENV HOSTS=mips64el-unknown-linux-gnuabi64

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
