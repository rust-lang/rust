FROM ubuntu:22.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh

WORKDIR /tmp

COPY scripts/crosstool-ng-build.sh /scripts/
COPY host-x86_64/dist-powerpc64-linux-musl/powerpc64-unknown-linux-musl.defconfig /tmp/crosstool.defconfig
RUN /scripts/crosstool-ng-build.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/powerpc64-unknown-linux-musl/bin

ENV \
    AR_powerpc64_unknown_linux_musl=powerpc64-unknown-linux-musl-ar \
    CC_powerpc64_unknown_linux_musl=powerpc64-unknown-linux-musl-gcc \
    CXX_powerpc64_unknown_linux_musl=powerpc64-unknown-linux-musl-g++

ENV HOSTS=powerpc64-unknown-linux-musl

ENV RUST_CONFIGURE_ARGS \
    --enable-extended \
    --enable-full-tools \
    --enable-profiler \
    --enable-sanitizers \
    --disable-docs \
    --set target.powerpc64-unknown-linux-musl.crt-static=false \
    --musl-root-powerpc64=/x-tools/powerpc64-unknown-linux-musl/powerpc64-unknown-linux-musl/sysroot/usr

ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
