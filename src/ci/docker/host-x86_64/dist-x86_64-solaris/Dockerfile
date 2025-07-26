FROM ubuntu:22.04

COPY scripts/cross-apt-packages.sh /tmp/
RUN bash /tmp/cross-apt-packages.sh

# Required gcc dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY scripts/shared.sh /tmp/
COPY scripts/solaris-toolchain.sh /tmp/

RUN bash /tmp/solaris-toolchain.sh x86_64 sysroot
RUN bash /tmp/solaris-toolchain.sh x86_64 binutils
RUN bash /tmp/solaris-toolchain.sh x86_64 gcc

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

ENV \
    AR_x86_64_pc_solaris=x86_64-solaris-ar \
    RANLIB_x86_64_pc_solaris=x86_64-solaris-ranlib \
    CC_x86_64_pc_solaris=x86_64-solaris-gcc \
    CXX_x86_64_pc_solaris=x86_64-solaris-g++

ENV HOSTS=x86_64-pc-solaris

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
