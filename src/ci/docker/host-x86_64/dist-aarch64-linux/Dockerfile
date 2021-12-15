FROM ubuntu:20.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates
WORKDIR /tmp
RUN curl -f https://curl.se/ca/cacert.pem -o cacert.pem

FROM ubuntu:16.04

# The ca-certificates in ubuntu-16 is too old, so update the certificates
# with something more recent.
COPY --from=0 /tmp/cacert.pem /tmp/cacert.pem
ENV CURL_CA_BUNDLE /tmp/cacert.pem

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

# Ubuntu 16.04 (this container) ships with make 4, but something in the
# toolchains we build below chokes on that, so go back to make 3
COPY scripts/make3.sh /scripts/
RUN sh /scripts/make3.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
USER rustbuild
WORKDIR /tmp

COPY host-x86_64/dist-aarch64-linux/aarch64-linux-gnu.config host-x86_64/dist-aarch64-linux/build-toolchains.sh /tmp/
RUN ./build-toolchains.sh

USER root

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

COPY scripts/cmake.sh /scripts/
RUN /scripts/cmake.sh

ENV PATH=$PATH:/x-tools/aarch64-unknown-linux-gnueabi/bin

ENV CC_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnueabi-gcc \
    AR_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnueabi-ar \
    CXX_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnueabi-g++

ENV HOSTS=aarch64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS \
      --enable-full-tools \
      --enable-profiler \
      --enable-sanitizers
ENV SCRIPT python3 ../x.py dist --host $HOSTS --target $HOSTS
