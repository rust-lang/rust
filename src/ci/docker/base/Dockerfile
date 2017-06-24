FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  automake \
  bc \
  bison \
  bzip2 \
  ca-certificates \
  cmake \
  cpio \
  curl \
  file \
  flex \
  g++ \
  gawk \
  gdb \
  git \
  gperf \
  help2man \
  libc6-dev \
  libedit-dev \
  libncurses-dev \
  libncurses5-dev \
  libssl-dev \
  libtool-bin \
  make \
  patch \
  pkg-config \
  python2.7 \
  python2.7-dev \
  sudo \
  texinfo \
  unzip \
  wget \
  xz-utils \
  zlib1g-dev

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-05-12-sccache-x86_64-unknown-linux-musl && \
      chmod +x /usr/local/bin/sccache

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]
