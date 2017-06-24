FROM rust-ci-base

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

COPY dist-armv7-linux/build-toolchains.sh dist-armv7-linux/armv7-linux-gnueabihf.config /tmp/
RUN ./build-toolchains.sh

USER root

ENV PATH=$PATH:/x-tools/armv7-unknown-linux-gnueabihf/bin

ENV CC_armv7_unknown_linux_gnueabihf=armv7-unknown-linux-gnueabihf-gcc \
    AR_armv7_unknown_linux_gnueabihf=armv7-unknown-linux-gnueabihf-ar \
    CXX_armv7_unknown_linux_gnueabihf=armv7-unknown-linux-gnueabihf-g++

ENV HOSTS=armv7-unknown-linux-gnueabihf

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
