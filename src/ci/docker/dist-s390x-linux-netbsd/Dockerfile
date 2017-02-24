FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  automake \
  bison \
  bzip2 \
  ca-certificates \
  cmake \
  curl \
  file \
  flex \
  g++ \
  gawk \
  gdb \
  git \
  gperf \
  help2man \
  libncurses-dev \
  libtool-bin \
  make \
  patch \
  python2.7 \
  sudo \
  texinfo \
  wget \
  xz-utils \
  libssl-dev \
  pkg-config

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-02-24-sccache-x86_64-unknown-linux-gnu && \
      chmod +x /usr/local/bin/sccache

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Ubuntu 16.04 (this contianer) ships with make 4, but something in the
# toolchains we build below chokes on that, so go back to make 3
RUN curl https://ftp.gnu.org/gnu/make/make-3.81.tar.gz | tar xzf - && \
      cd make-3.81 && \
      ./configure --prefix=/usr && \
      make && \
      make install && \
      cd .. && \
      rm -rf make-3.81

RUN curl http://crosstool-ng.org/download/crosstool-ng/crosstool-ng-1.22.0.tar.bz2 | \
      tar xjf - && \
      cd crosstool-ng && \
      ./configure --prefix=/usr/local && \
      make -j$(nproc) && \
      make install && \
      cd .. && \
      rm -rf crosstool-ng

RUN groupadd -r rustbuild && useradd -m -r -g rustbuild rustbuild
RUN mkdir /x-tools && chown rustbuild:rustbuild /x-tools
USER rustbuild
WORKDIR /tmp

COPY patches/ /tmp/patches/
COPY s390x-linux-gnu.config build-s390x-toolchain.sh /tmp/
RUN ./build-s390x-toolchain.sh

USER root

COPY build-netbsd-toolchain.sh /tmp/
RUN ./build-netbsd-toolchain.sh

ENV PATH=$PATH:/x-tools/s390x-ibm-linux-gnu/bin

ENV \
    AR_x86_64_unknown_netbsd=x86_64-unknown-netbsd-ar \
    CC_x86_64_unknown_netbsd=x86_64-unknown-netbsd-gcc \
    CXX_x86_64_unknown_netbsd=x86_64-unknown-netbsd-g++ \
    CC_s390x_unknown_linux_gnu=s390x-ibm-linux-gnu-gcc \
    AR_s390x_unknown_linux_gnu=s390x-ibm-linux-gnu-ar \
    CXX_s390x_unknown_linux_gnu=s390x-ibm-linux-gnu-g++

ENV HOSTS=x86_64-unknown-netbsd
ENV HOSTS=$HOSTS,s390x-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
