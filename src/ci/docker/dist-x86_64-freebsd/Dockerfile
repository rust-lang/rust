FROM rust-ci-base

COPY dist-x86_64-freebsd/build-toolchain.sh /tmp/
RUN /tmp/build-toolchain.sh x86_64

ENV \
    AR_x86_64_unknown_freebsd=x86_64-unknown-freebsd10-ar \
    CC_x86_64_unknown_freebsd=x86_64-unknown-freebsd10-gcc \
    CXX_x86_64_unknown_freebsd=x86_64-unknown-freebsd10-g++

ENV HOSTS=x86_64-unknown-freebsd

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
