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
  pkg-config \
  software-properties-common \
  apt-transport-https

# Install the cross compiler
RUN apt-key adv --batch --yes --keyserver keyserver.ubuntu.com --recv-keys AA12E97F0881517F && \
    add-apt-repository -y 'deb https://static.redox-os.org/toolchain/apt /' && \
    apt-get update && \
    apt-get install -y x86-64-unknown-redox-gcc

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-05-12-sccache-x86_64-unknown-linux-musl && \
      chmod +x /usr/local/bin/sccache

ENV \
    AR_x86_64_unknown_redox=x86_64-unknown-redox-ar \
    CC_x86_64_unknown_redox=x86_64-unknown-redox-gcc \
    CXX_x86_64_unknown_redox=x86_64-unknown-redox-g++

ENV RUST_CONFIGURE_ARGS --target=x86_64-unknown-redox --enable-extended
ENV SCRIPT python2.7 ../x.py dist --target x86_64-unknown-redox
