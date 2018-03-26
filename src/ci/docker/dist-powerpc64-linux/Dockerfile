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

COPY dist-powerpc64-linux/patches/ /tmp/patches/
COPY dist-powerpc64-linux/shared.sh dist-powerpc64-linux/powerpc64-linux-gnu.config dist-powerpc64-linux/build-powerpc64-toolchain.sh /tmp/
RUN ./build-powerpc64-toolchain.sh

USER root

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/powerpc64-unknown-linux-gnu/bin

ENV \
    AR_powerpc64_unknown_linux_gnu=powerpc64-unknown-linux-gnu-ar \
    CC_powerpc64_unknown_linux_gnu=powerpc64-unknown-linux-gnu-gcc \
    CXX_powerpc64_unknown_linux_gnu=powerpc64-unknown-linux-gnu-g++

ENV HOSTS=powerpc64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
