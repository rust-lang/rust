FROM ubuntu:17.10

ENV TARGETS=aarch64-unknown-cloudabi
# FIXME(EdSchouten): Enable ARMv7 support once libc ≥0.2.37 has been merged.
# ENV TARGETS=armv7-unknown-cloudabi-eabihf
ENV TARGETS=$TARGETS,i686-unknown-cloudabi
ENV TARGETS=$TARGETS,x86_64-unknown-cloudabi

COPY scripts/cloudabi-toolchain.sh /tmp/
RUN /tmp/cloudabi-toolchain.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

# FIXME(EdSchouten): Remove this once cc ≥1.0.4 has been merged. It can
# automatically pick the right compiler path.
ENV \
    AR_aarch64_unknown_cloudabi=aarch64-unknown-cloudabi-ar \
    CC_aarch64_unknown_cloudabi=aarch64-unknown-cloudabi-clang \
    CXX_aarch64_unknown_cloudabi=aarch64-unknown-cloudabi-clang++ \
    AR_i686_unknown_cloudabi=i686-unknown-cloudabi-ar \
    CC_i686_unknown_cloudabi=i686-unknown-cloudabi-clang \
    CXX_i686_unknown_cloudabi=i686-unknown-cloudabi-clang++ \
    AR_x86_64_unknown_cloudabi=x86_64-unknown-cloudabi-ar \
    CC_x86_64_unknown_cloudabi=x86_64-unknown-cloudabi-clang \
    CXX_x86_64_unknown_cloudabi=x86_64-unknown-cloudabi-clang++

# FIXME(EdSchouten): Work towards being able to run 'x.py test'.
ENV RUST_CONFIGURE_ARGS --target=${TARGETS} --disable-jemalloc
ENV SCRIPT python2.7 /checkout/x.py dist --target ${TARGETS}
