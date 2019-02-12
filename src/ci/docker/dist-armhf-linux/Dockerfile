FROM ubuntu:16.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

# Ubuntu 16.04 (this container) ships with make 4, but something in the
# toolchains we build below chokes on that, so go back to make 3
COPY scripts/make3.sh /scripts/
RUN sh /scripts/make3.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
USER rustbuild
WORKDIR /tmp

COPY dist-armhf-linux/patches/ /tmp/patches/
COPY dist-armhf-linux/arm-linux-gnueabihf.config dist-armhf-linux/build-toolchains.sh /tmp/
RUN ./build-toolchains.sh

USER root

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/arm-unknown-linux-gnueabihf/bin

ENV CC_arm_unknown_linux_gnueabihf=arm-unknown-linux-gnueabihf-gcc \
    AR_arm_unknown_linux_gnueabihf=arm-unknown-linux-gnueabihf-ar \
    CXX_arm_unknown_linux_gnueabihf=arm-unknown-linux-gnueabihf-g++

ENV HOSTS=arm-unknown-linux-gnueabihf

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
