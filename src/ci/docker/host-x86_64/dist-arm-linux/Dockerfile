FROM ubuntu:20.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng-1.24.sh /scripts/
RUN sh /scripts/crosstool-ng-1.24.sh

WORKDIR /build

COPY scripts/musl-patch-configure.diff /build/
COPY scripts/musl-toolchain.sh /build/
# We need to mitigate rust-lang/rust#34978 when compiling musl itself as well
RUN CFLAGS="-Wa,--compress-debug-sections=none -Wl,--compress-debug-sections=none" \
    CXXFLAGS="-Wa,--compress-debug-sections=none -Wl,--compress-debug-sections=none" \
    bash musl-toolchain.sh aarch64 && rm -rf build

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
USER rustbuild
WORKDIR /tmp

COPY host-x86_64/dist-arm-linux/arm-linux-gnueabi.config host-x86_64/dist-arm-linux/build-toolchains.sh /tmp/
RUN ./build-toolchains.sh

USER root

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

ENV PATH=$PATH:/x-tools/arm-unknown-linux-gnueabi/bin

ENV CC_arm_unknown_linux_gnueabi=arm-unknown-linux-gnueabi-gcc \
    AR_arm_unknown_linux_gnueabi=arm-unknown-linux-gnueabi-ar \
    CXX_arm_unknown_linux_gnueabi=arm-unknown-linux-gnueabi-g++

ENV HOSTS=arm-unknown-linux-gnueabi,aarch64-unknown-linux-musl

ENV RUST_CONFIGURE_ARGS --enable-full-tools --disable-docs --musl-root-aarch64=/usr/local/aarch64-linux-musl \
    --set target.aarch64-unknown-linux-musl.crt-static=false
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
