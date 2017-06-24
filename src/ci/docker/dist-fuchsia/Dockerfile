FROM rust-ci-base

RUN apt-get update && apt-get install -y --no-install-recommends \
  ninja-build \
  swig

RUN curl -L https://cmake.org/files/v3.8/cmake-3.8.0-rc1-Linux-x86_64.tar.gz | \
      tar xzf - -C /usr/local --strip-components=1

WORKDIR /tmp
COPY dist-fuchsia/shared.sh dist-fuchsia/build-toolchain.sh dist-fuchsia/compiler-rt-dso-handle.patch /tmp/
RUN /tmp/build-toolchain.sh

ENV \
    AR_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-ar \
    CC_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang \
    CXX_x86_64_unknown_fuchsia=x86_64-unknown-fuchsia-clang++ \
    AR_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-ar \
    CC_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang \
    CXX_aarch64_unknown_fuchsia=aarch64-unknown-fuchsia-clang++

ENV TARGETS=x86_64-unknown-fuchsia
ENV TARGETS=$TARGETS,aarch64-unknown-fuchsia

ENV RUST_CONFIGURE_ARGS --target=$TARGETS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --target $TARGETS
