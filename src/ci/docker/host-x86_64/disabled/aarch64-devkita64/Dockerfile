FROM devkitpro/devkita64:20200528

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
USER rustbuild
WORKDIR /tmp

USER root

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PATH=$PATH:/opt/devkitpro/devkitA64/bin/

ENV CC_aarch64_unknown_switch_devkita64=aarch64-none-elf-gcc \
    AR_aarch64_unknown_switch_devkita64=aarch64-none-elf-ar \
    CXX_aarch64_unknown_switch_devkita64=aarch64-none-elf-g++

ENV TARGETS=aarch64-unknown-switch-devkita64

ENV RUST_CONFIGURE_ARGS \
      --disable-docs
ENV SCRIPT python3 ../x.py dist --target $TARGETS
