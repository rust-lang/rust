FROM ubuntu:17.10

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

RUN apt-get build-dep -y clang llvm && apt-get install -y --no-install-recommends \
  gcc-arm-none-eabi \
  libnewlib-arm-none-eabi

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV TARGETS=thumbv6m-none-eabi
ENV TARGETS=$TARGETS,thumbv7m-none-eabi
ENV TARGETS=$TARGETS,thumbv7em-none-eabi
ENV TARGETS=$TARGETS,thumbv7em-none-eabihf

ENV RUST_CONFIGURE_ARGS --disable-docs
ENV SCRIPT python2.7 ../x.py dist --target $TARGETS
