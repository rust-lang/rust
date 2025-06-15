FROM ubuntu:22.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
WORKDIR /tmp

COPY scripts/crosstool-ng-build.sh /scripts/
COPY host-x86_64/dist-armv7-linux-musl/armv7-unknown-linux-musleabihf.defconfig /tmp/crosstool.defconfig
RUN /scripts/crosstool-ng-build.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/armv7-unknown-linux-musleabihf/bin

ENV CC_armv7_unknown_linux_musleabihf=armv7-unknown-linux-musleabihf-gcc \
    AR_armv7_unknown_linux_musleabihf=armv7-unknown-linux-musleabihf-ar \
    CXX_armv7_unknown_linux_musleabihf=armv7-unknown-linux-musleabihf-g++

ENV HOSTS=armv7-unknown-linux-musleabihf

ENV RUST_CONFIGURE_ARGS \
      --enable-extended \
      --enable-full-tools \
      --enable-profiler \
      --enable-sanitizers \
      --disable-docs \
      --set target.armv7-unknown-linux-musleabihf.crt-static=false \
      --musl-root-armv7hf=/x-tools/armv7-unknown-linux-musleabihf/armv7-unknown-linux-musleabihf/sysroot/usr

ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
