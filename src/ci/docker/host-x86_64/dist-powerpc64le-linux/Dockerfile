FROM ubuntu:20.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
USER rustbuild
WORKDIR /tmp

USER root

RUN apt-get install -y --no-install-recommends rpm2cpio cpio
COPY host-x86_64/dist-powerpc64le-linux/shared.sh host-x86_64/dist-powerpc64le-linux/build-powerpc64le-toolchain.sh /tmp/
RUN ./build-powerpc64le-toolchain.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-ar \
    CC_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-gcc \
    CXX_powerpc64le_unknown_linux_gnu=powerpc64le-linux-gnu-g++

ENV HOSTS=powerpc64le-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --enable-extended --enable-profiler --disable-docs
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
