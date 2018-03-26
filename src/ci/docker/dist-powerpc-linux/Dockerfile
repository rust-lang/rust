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

COPY dist-powerpc-linux/patches/ /tmp/patches/
COPY dist-powerpc-linux/powerpc-linux-gnu.config dist-powerpc-linux/build-powerpc-toolchain.sh /tmp/
RUN ./build-powerpc-toolchain.sh

USER root

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/powerpc-unknown-linux-gnu/bin

ENV \
    CC_powerpc_unknown_linux_gnu=powerpc-unknown-linux-gnu-gcc \
    AR_powerpc_unknown_linux_gnu=powerpc-unknown-linux-gnu-ar \
    CXX_powerpc_unknown_linux_gnu=powerpc-unknown-linux-gnu-g++

ENV HOSTS=powerpc-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS

# FIXME(#36150) this will fail the bootstrap. Probably means something bad is
#               happening!
ENV NO_LLVM_ASSERTIONS 1
