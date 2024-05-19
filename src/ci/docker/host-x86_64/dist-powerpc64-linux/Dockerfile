FROM ubuntu:22.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
WORKDIR /tmp

COPY scripts/crosstool-ng-build.sh /scripts/
COPY host-x86_64/dist-powerpc64-linux/powerpc64-linux-gnu.defconfig /tmp/crosstool.defconfig
RUN /scripts/crosstool-ng-build.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/powerpc64-unknown-linux-gnu/bin

ENV \
    AR_powerpc64_unknown_linux_gnu=powerpc64-unknown-linux-gnu-ar \
    CC_powerpc64_unknown_linux_gnu=powerpc64-unknown-linux-gnu-gcc \
    CXX_powerpc64_unknown_linux_gnu=powerpc64-unknown-linux-gnu-g++

ENV HOSTS=powerpc64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --enable-extended --enable-profiler --disable-docs
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
