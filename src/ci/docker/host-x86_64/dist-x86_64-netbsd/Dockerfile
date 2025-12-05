FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh
RUN apt-get install -y zlib1g-dev

COPY host-x86_64/dist-x86_64-netbsd/build-netbsd-toolchain.sh /tmp/
# GCC 10 changed the default to -fno-common, which causes errors with the NetBSD-9.0 sources like:
# /usr/bin/ld: buf.o:(.bss+0x0): multiple definition of `debug_file'; arch.o:(.bss+0x0): first defined here
RUN env HOST_CFLAGS="-O -fcommon" /tmp/build-netbsd-toolchain.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/x86_64-unknown-netbsd/bin

ENV \
    AR_x86_64_unknown_netbsd=x86_64--netbsd-ar \
    CC_x86_64_unknown_netbsd=x86_64--netbsd-gcc-sysroot \
    CXX_x86_64_unknown_netbsd=x86_64--netbsd-g++-sysroot

ENV HOSTS=x86_64-unknown-netbsd

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
