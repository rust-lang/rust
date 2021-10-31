# We need recent curl, OpenSSL and CA certificates, so we can download further
# dependencies in the debian:6 image. We use an ubuntu 20.04 image download
# those.
FROM ubuntu:20.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates
WORKDIR /tmp
COPY host-x86_64/dist-x86_64-linux/download-openssl-curl.sh /tmp/
RUN ./download-openssl-curl.sh

# We use Debian 6 (glibc 2.11, kernel 2.6.32) as a common base for other
# distros that still need Rust support: RHEL 6 (glibc 2.12, kernel 2.6.32) and
# SLES 11 SP4 (glibc 2.11, kernel 3.0).
FROM debian:6

WORKDIR /build

# Debian 6 is EOL and no longer available from the usual mirrors,
# so we'll need to switch to http://archive.debian.org/
RUN sed -i '/updates/d' /etc/apt/sources.list && \
    sed -i 's/httpredir/archive/' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install --allow-unauthenticated -y --no-install-recommends \
      automake \
      bzip2 \
      file \
      g++ \
      g++-multilib \
      gcc \
      gcc-multilib \
      git \
      lib32z1-dev \
      libedit-dev \
      libncurses-dev \
      make \
      patch \
      perl \
      pkg-config \
      unzip \
      wget \
      xz-utils \
      zlib1g-dev

ENV PATH=/rustroot/bin:$PATH
ENV LD_LIBRARY_PATH=/rustroot/lib64:/rustroot/lib32:/rustroot/lib
ENV PKG_CONFIG_PATH=/rustroot/lib/pkgconfig
WORKDIR /tmp
RUN mkdir /home/user
COPY host-x86_64/dist-x86_64-linux/shared.sh /tmp/

# We need a build of openssl which supports SNI to download artifacts from
# static.rust-lang.org. This'll be used to link into libcurl below (and used
# later as well), so build a copy of OpenSSL with dynamic libraries into our
# generic root.
COPY --from=0 /tmp/openssl.tar.gz /tmp/openssl.tar.gz
COPY host-x86_64/dist-x86_64-linux/build-openssl.sh /tmp/
RUN ./build-openssl.sh

# The `curl` binary on Debian 6 doesn't support SNI which is needed for fetching
# some https urls we have, so install a new version of libcurl + curl which is
# using the openssl we just built previously.
#
# Note that we also disable a bunch of optional features of curl that we don't
# really need.
COPY --from=0 /tmp/curl.tar.xz /tmp/curl.tar.xz
COPY host-x86_64/dist-x86_64-linux/build-curl.sh /tmp/
RUN ./build-curl.sh

# Use up-to-date curl CA bundle
COPY --from=0 /tmp/cacert.pem /tmp/cacert.pem
ENV CURL_CA_BUNDLE /tmp/cacert.pem

# binutils < 2.22 has a bug where the 32-bit executables it generates
# immediately segfault in Rust, so we need to install our own binutils.
#
# See https://github.com/rust-lang/rust/issues/20440 for more info
COPY host-x86_64/dist-x86_64-linux/build-binutils.sh /tmp/
RUN ./build-binutils.sh

# Need at least GCC 5.1 to compile LLVM nowadays
COPY host-x86_64/dist-x86_64-linux/build-gcc.sh /tmp/
RUN ./build-gcc.sh && apt-get remove -y gcc g++

COPY host-x86_64/dist-x86_64-linux/build-python.sh /tmp/
# Build Python 3 needed for LLVM 12.
RUN ./build-python.sh 3.9.1

# LLVM needs cmake 3.13.4 or higher.
COPY host-x86_64/dist-x86_64-linux/build-cmake.sh /tmp/
RUN ./build-cmake.sh

# Now build LLVM+Clang, afterwards configuring further compilations to use the
# clang/clang++ compilers.
COPY host-x86_64/dist-x86_64-linux/build-clang.sh /tmp/
RUN ./build-clang.sh
ENV CC=clang CXX=clang++

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV HOSTS=i686-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS \
      --enable-full-tools \
      --enable-sanitizers \
      --enable-profiler \
      --set target.i686-unknown-linux-gnu.linker=clang \
      --build=i686-unknown-linux-gnu \
      --set llvm.ninja=false \
      --set rust.jemalloc
ENV SCRIPT python3 ../x.py dist --build $HOSTS --host $HOSTS --target $HOSTS
ENV CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_LINKER=clang

# This was added when we switched from gcc to clang. It's not clear why this is
# needed unfortunately, but without this the stage1 bootstrap segfaults
# somewhere inside of a build script. The build ends up just hanging instead of
# actually killing the process that segfaulted, but if the process is run
# manually in a debugger the segfault is immediately seen as well as the
# misaligned stack access.
#
# Added in #50200 there's some more logs there
ENV CFLAGS -mstackrealign

# When we build cargo in this container, we don't want it to use the system
# libcurl, instead it should compile its own.
ENV LIBCURL_NO_PKG_CONFIG 1

# There was a bad interaction between "old" 32-bit binaries on current 64-bit
# kernels with selinux enabled, where ASLR mmap would sometimes choose a low
# address and then block it for being below `vm.mmap_min_addr` -> `EACCES`.
# This is probably a kernel bug, but setting `ulimit -Hs` works around it.
# See also `src/ci/run.sh` where this takes effect.
ENV SET_HARD_RLIMIT_STACK 1

ENV DIST_REQUIRE_ALL_TOOLS 1
