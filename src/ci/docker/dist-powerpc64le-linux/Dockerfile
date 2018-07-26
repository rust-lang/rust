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

USER root

RUN apt-get install -y --no-install-recommends rpm2cpio cpio
COPY dist-powerpc64le-linux/shared.sh dist-powerpc64le-linux/build-powerpc64le-toolchain.sh /tmp/
RUN ./build-powerpc64le-toolchain.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-ar \
    CC_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-gcc \
    CXX_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-g++

ENV HOSTS=powerpc64le-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
