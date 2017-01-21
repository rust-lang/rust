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
  xz-utils

ENV SCCACHE_DIGEST=7237e38e029342fa27b7ac25412cb9d52554008b12389727320bd533fd7f05b6a96d55485f305caf95e5c8f5f97c3313e10012ccad3e752aba2518f3522ba783
RUN curl -L https://api.pub.build.mozilla.org/tooltool/sha512/$SCCACHE_DIGEST | \
      tar xJf - -C /usr/local/bin --strip-components=1

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

COPY arm-linux-gnueabihf.config arm-linux-gnueabi.config build-toolchains.sh /tmp/
RUN ./build-toolchains.sh

USER root

ENV PATH=$PATH:/x-tools/arm-unknown-linux-gnueabi/bin
ENV PATH=$PATH:/x-tools/arm-unknown-linux-gnueabihf/bin

ENV CC_arm_unknown_linux_gnueabi=arm-unknown-linux-gnueabi-gcc \
    AR_arm_unknown_linux_gnueabi=arm-unknown-linux-gnueabi-ar \
    CXX_arm_unknown_linux_gnueabi=arm-unknown-linux-gnueabi-g++ \
    CC_arm_unknown_linux_gnueabihf=arm-unknown-linux-gnueabihf-gcc \
    AR_arm_unknown_linux_gnueabihf=arm-unknown-linux-gnueabihf-ar \
    CXX_arm_unknown_linux_gnueabihf=arm-unknown-linux-gnueabihf-g++

ENV HOSTS=arm-unknown-linux-gnueabi
ENV HOSTS=$HOSTS,arm-unknown-linux-gnueabihf

ENV RUST_CONFIGURE_ARGS --host=$HOSTS
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
