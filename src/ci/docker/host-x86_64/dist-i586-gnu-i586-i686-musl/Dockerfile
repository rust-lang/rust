FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  g++-multilib \
  make \
  ninja-build \
  file \
  curl \
  ca-certificates \
  python3 \
  git \
  cmake \
  xz-utils \
  sudo \
  gdb \
  patch \
  libssl-dev \
  pkg-config

COPY scripts/cross-apt-packages.sh /scripts/
RUN sh /scripts/cross-apt-packages.sh

COPY scripts/crosstool-ng.sh /scripts/
RUN sh /scripts/crosstool-ng.sh

COPY scripts/rustbuild-setup.sh /scripts/
RUN sh /scripts/rustbuild-setup.sh
WORKDIR /tmp

COPY scripts/crosstool-ng-build.sh /scripts/
COPY host-x86_64/dist-i586-gnu-i586-i686-musl/i586-linux-gnu.defconfig /tmp/crosstool.defconfig
RUN /scripts/crosstool-ng-build.sh

ENV PATH=$PATH:/x-tools/i586-unknown-linux-gnu/bin
ENV \
    CC_i586_unknown_linux_gnu=i586-unknown-linux-gnu-gcc \
    AR_i586_unknown_linux_gnu=i586-unknown-linux-gnu-ar

WORKDIR /build/
COPY scripts/musl.sh /build/
RUN CC=gcc CFLAGS="-m32 -Wa,-mrelax-relocations=no" \
    CXX=g++ CXXFLAGS="-m32 -Wa,-mrelax-relocations=no" \
    bash musl.sh i686 --target=i686 && \
    CC=gcc CFLAGS="-march=pentium -m32 -Wa,-mrelax-relocations=no" \
    CXX=g++ CXXFLAGS="-march=pentium -m32 -Wa,-mrelax-relocations=no" \
    bash musl.sh i586 --target=i586 && \
    rm -rf /build

# FIXME: musl really shouldn't be linking libgcc_s.so, as it's linked to glibc,
# but it's required by tests/ui/proc-macro/crt-static.rs. Ubuntu:16.04 gcc-5
# had libgcc_s.so as a symlink to the absolute libgcc_s.so.1, but now it's an
# ld-script that expects to find libgcc_s.so.1 in the library search path.
# See also https://github.com/rust-lang/rust/issues/82521
RUN ln -s /usr/lib32/libgcc_s.so.1 /musl-i686/lib/

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV RUST_CONFIGURE_ARGS \
      --musl-root-i586=/musl-i586 \
      --musl-root-i686=/musl-i686 \
      --disable-docs

# Newer binutils broke things on some vms/distros (i.e., linking against
# unknown relocs disabled by the following flag), so we need to go out of our
# way to produce "super compatible" binaries.
#
# See: https://github.com/rust-lang/rust/issues/34978
ENV CFLAGS_i686_unknown_linux_musl=-Wa,-mrelax-relocations=no
ENV CFLAGS_i586_unknown_linux_gnu=-Wa,-mrelax-relocations=no
ENV CFLAGS_i586_unknown_linux_musl=-Wa,-mrelax-relocations=no

ENV TARGETS=i586-unknown-linux-gnu,i686-unknown-linux-musl

ENV SCRIPT \
      python3 ../x.py --stage 2 test --host='' --target $TARGETS && \
      python3 ../x.py dist --host='' --target $TARGETS,i586-unknown-linux-musl
