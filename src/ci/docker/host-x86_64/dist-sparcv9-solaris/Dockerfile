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

RUN bash /tmp/solaris-toolchain.sh sparcv9 sysroot
RUN bash /tmp/solaris-toolchain.sh sparcv9 binutils
RUN bash /tmp/solaris-toolchain.sh sparcv9 gcc

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

ENV \
    AR_sparcv9_sun_solaris=sparcv9-solaris-ar \
    RANLIB_sparcv9_sun_solaris=sparcv9-solaris-ranlib \
    CC_sparcv9_sun_solaris=sparcv9-solaris-gcc \
    CXX_sparcv9_sun_solaris=sparcv9-solaris-g++

ENV HOSTS=sparcv9-sun-solaris

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
