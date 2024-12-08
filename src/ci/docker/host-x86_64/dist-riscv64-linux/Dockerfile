FROM ubuntu:22.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
WORKDIR /tmp

COPY scripts/crosstool-ng-build.sh /scripts/
COPY host-x86_64/dist-riscv64-linux/patches/ /tmp/patches/
COPY host-x86_64/dist-riscv64-linux/riscv64-unknown-linux-gnu.defconfig /tmp/crosstool.defconfig
RUN /scripts/crosstool-ng-build.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/riscv64-unknown-linux-gnu/bin

ENV CC_riscv64gc_unknown_linux_gnu=riscv64-unknown-linux-gnu-gcc \
    AR_riscv64gc_unknown_linux_gnu=riscv64-unknown-linux-gnu-ar \
    CXX_riscv64gc_unknown_linux_gnu=riscv64-unknown-linux-gnu-g++

ENV HOSTS=riscv64gc-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --enable-extended --enable-profiler --disable-docs
ENV SCRIPT python3 ../x.py dist --target $HOSTS --host $HOSTS
