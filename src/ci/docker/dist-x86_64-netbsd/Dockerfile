FROM ubuntu:16.04

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY dist-x86_64-netbsd/build-netbsd-toolchain.sh /tmp/
RUN /tmp/build-netbsd-toolchain.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/x-tools/x86_64-unknown-netbsd/bin

ENV \
    AR_x86_64_unknown_netbsd=x86_64--netbsd-ar \
    CC_x86_64_unknown_netbsd=x86_64--netbsd-gcc-sysroot \
    CXX_x86_64_unknown_netbsd=x86_64--netbsd-g++-sysroot

ENV HOSTS=x86_64-unknown-netbsd

ENV RUST_CONFIGURE_ARGS --enable-extended --disable-docs
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
