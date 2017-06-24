FROM rust-ci-base

# Ubuntu 16.04 (this contianer) ships with make 4, but something in the
# toolchains we build below chokes on that, so go back to make 3
RUN curl https://ftp.gnu.org/gnu/make/make-3.81.tar.gz | tar xzf - && \
      cd make-3.81 && \
      ./configure --prefix=/usr && \
      make && \
      make install && \
      cd .. && \
      rm -rf make-3.81

RUN curl http://crosstool-ng.org/download/crosstool-ng/crosstool-ng-1.22.0.tar.bz2 | \
      tar xjf - && \
      cd crosstool-ng && \
      ./configure --prefix=/usr/local && \
      make -j$(nproc) && \
      make install && \
      cd .. && \
      rm -rf crosstool-ng

RUN groupadd -r rustbuild && useradd -m -r -g rustbuild rustbuild
RUN mkdir /x-tools && chown rustbuild:rustbuild /x-tools
USER rustbuild
WORKDIR /tmp

COPY dist-powerpc64-linux/patches/ /tmp/patches/
COPY dist-powerpc64-linux/shared.sh dist-powerpc64-linux/powerpc64-linux-gnu.config dist-powerpc64-linux/build-powerpc64-toolchain.sh /tmp/
RUN ./build-powerpc64-toolchain.sh

USER root

ENV PATH=$PATH:/x-tools/powerpc64-unknown-linux-gnu/bin

ENV \
    AR_powerpc64_unknown_linux_gnu=powerpc64-unknown-linux-gnu-ar \
    CC_powerpc64_unknown_linux_gnu=powerpc64-unknown-linux-gnu-gcc \
    CXX_powerpc64_unknown_linux_gnu=powerpc64-unknown-linux-gnu-g++

ENV HOSTS=powerpc64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS --host=$HOSTS --enable-extended
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS
