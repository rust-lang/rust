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

COPY dist-x86_64-netbsd/build-netbsd-toolchain.sh /tmp/
RUN ./build-netbsd-toolchain.sh

USER root

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/x86_64-unknown-netbsd/bin

ENV \
    AR_x86_64_unknown_netbsd=x86_64--netbsd-ar \
    CC_x86_64_unknown_netbsd=x86_64--netbsd-gcc-sysroot \
    CXX_x86_64_unknown_netbsd=x86_64--netbsd-g++-sysroot

ENV HOSTS=x86_64-unknown-netbsd

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs \
  --set llvm.allow-old-toolchain
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
