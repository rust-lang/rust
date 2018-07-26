FROM ubuntu:17.10

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

RUN apt-get build-dep -y clang llvm && apt-get install -y --no-install-recommends \
  build-essential \
  gcc-multilib \
  libedit-dev \
  libgmp-dev \
  libisl-dev \
  libmpc-dev \
  libmpfr-dev \
  ninja-build \
  nodejs \
  python2.7-dev \
  software-properties-common \
  unzip

RUN apt-key adv --batch --yes --keyserver keyserver.ubuntu.com --recv-keys 74DA7924C5513486
RUN add-apt-repository -y 'deb http://apt.dilos.org/dilos dilos2-testing main'

WORKDIR /tmp
COPY dist-various-2/shared.sh /tmp/
COPY dist-various-2/build-cloudabi-toolchain.sh /tmp/
RUN /tmp/build-cloudabi-toolchain.sh x86_64-unknown-cloudabi
COPY dist-various-2/build-fuchsia-toolchain.sh /tmp/
RUN /tmp/build-fuchsia-toolchain.sh
COPY dist-various-2/build-solaris-toolchain.sh /tmp/
RUN /tmp/build-solaris-toolchain.sh x86_64  amd64   solaris-i386
RUN /tmp/build-solaris-toolchain.sh sparcv9 sparcv9 solaris-sparc

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-ar \
    CC_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang \
    CXX_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang++ \
    AR_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-ar \
    CC_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang \
    CXX_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang++ \
    AR_sparcv9_sun_solaris=sparcv9-sun-solaris2.10-ar \
    CC_sparcv9_sun_solaris=sparcv9-sun-solaris2.10-gcc \
    CXX_sparcv9_sun_solaris=sparcv9-sun-solaris2.10-g++ \
    AR_x86_64_sun_solaris=x86_64-sun-solaris2.10-ar \
    CC_x86_64_sun_solaris=x86_64-sun-solaris2.10-gcc \
    CXX_x86_64_sun_solaris=x86_64-sun-solaris2.10-g++

ENV TARGETS=x86_64-unknown-fuchsia
ENV TARGETS=$TARGETS,aarch64-unknown-fuchsia
ENV TARGETS=$TARGETS,sparcv9-sun-solaris
ENV TARGETS=$TARGETS,wasm32-unknown-unknown
ENV TARGETS=$TARGETS,x86_64-sun-solaris
ENV TARGETS=$TARGETS,x86_64-unknown-linux-gnux32
ENV TARGETS=$TARGETS,x86_64-unknown-cloudabi

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python2.7 ../x.py dist --target $TARGETS
