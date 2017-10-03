FROM ubuntu:16.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

RUN apt-get install -y --no-install-recommends \
                    software-properties-common libgmp-dev libmpfr-dev libmpc-dev libisl-dev
RUN apt-key adv --batch --yes --keyserver keyserver.ubuntu.com --recv-keys 74DA7924C5513486
RUN add-apt-repository -y 'deb http://apt.dilos.org/dilos dilos2-testing main'

COPY dist-solaris/build-toolchain.sh /tmp/
RUN /tmp/build-toolchain.sh x86_64  amd64   solaris-i386
RUN /tmp/build-toolchain.sh sparcv9 sparcv9 solaris-sparc

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV \
    AR_sparcv9_sun_solaris=sparcv9-sun-solaris2.11-ar      \
    CC_sparcv9_sun_solaris=sparcv9-sun-solaris2.11-sysroot \
    CXX_sparcv9_sun_solaris=sparcv9-sun-solaris2.11-g++    \
    AR_x86_64_sun_solaris=x86_64-sun-solaris2.11-ar        \
    CC_x86_64_sun_solaris=x86_64-sun-solaris2.11-sysroot   \
    CXX_x86_64_sun_solaris=x86_64-sun-solaris2.11-g++

ENV TARGETS=sparcv9-sun-solaris,x86_64-sun-solaris

ENV RUST_CONFIGURE_ARGS --target=$TARGETS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --target $TARGETS
