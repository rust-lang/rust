FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  sudo \
  bzip2 \
  xz-utils \
  wget \
  libssl-dev \
  pkg-config

COPY dist-i686-freebsd/build-toolchain.sh /tmp/
RUN /tmp/build-toolchain.sh i686

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-05-12-sccache-x86_64-unknown-linux-musl && \
      chmod +x /usr/local/bin/sccache

ENV \
    AR_i686_unknown_freebsd=i686-unknown-freebsd10-ar \
    CC_i686_unknown_freebsd=i686-unknown-freebsd10-gcc \
    CXX_i686_unknown_freebsd=i686-unknown-freebsd10-g++

ENV HOSTS=i686-unknown-freebsd

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
