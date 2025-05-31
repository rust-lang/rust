# We document platform support for minimum glibc 2.17 and kernel 3.2.
# CentOS 7 has headers for kernel 3.10, but that's fine as long as we don't
# actually use newer APIs in rustc or std without a fallback. It's more
# important that we match glibc for ELF symbol versioning.
FROM ghcr.io/rust-lang/centos:7

WORKDIR /build

# CentOS 7 EOL is June 30, 2024, but the repos remain in the vault.
RUN sed -i /etc/yum.repos.d/*.repo -e 's!^mirrorlist!#mirrorlist!' \
  -e 's!^#baseurl=http://mirror.centos.org/!baseurl=https://vault.centos.org/!'
RUN sed -i 's/enabled=1/enabled=0/' /etc/yum/pluginconf.d/fastestmirror.conf

RUN yum upgrade -y && \
    yum install -y \
      automake \
      bzip2 \
      file \
      gcc \
      gcc-c++ \
      git \
      glibc-devel.i686 \
      glibc-devel.x86_64 \
      glibc-static.i686 \
      glibc-static.x86_64 \
      libedit-devel \
      libstdc++-devel.i686 \
      libstdc++-devel.x86_64 \
      make \
      ncurses-devel \
      openssl-devel \
      patch \
      perl \
      perl-core \
      pkgconfig \
      python3 \
      unzip \
      wget \
      flex \
      xz \
      zlib-devel.i686 \
      zlib-devel.x86_64 \
      && yum clean all

RUN mkdir -p /rustroot/bin

ENV PATH=/rustroot/bin:$PATH
ENV LD_LIBRARY_PATH=/rustroot/lib64:/rustroot/lib32:/rustroot/lib
ENV PKG_CONFIG_PATH=/rustroot/lib/pkgconfig
# Clang needs to access GCC headers to enable linker plugin LTO
WORKDIR /tmp
RUN mkdir /home/user
COPY scripts/shared.sh /tmp/

# Need at least GCC 5.1 to compile LLVM nowadays
COPY scripts/build-gcc.sh /tmp/
ENV GCC_VERSION=9.5.0
ENV GCC_BUILD_TARGET=x86_64-pc-linux-gnu
RUN ./build-gcc.sh && yum remove -y gcc gcc-c++

# LLVM 17 needs cmake 3.20 or higher.
COPY scripts/cmake.sh /tmp/
RUN ./cmake.sh

# Now build LLVM+Clang, afterwards configuring further compilations to use the
# clang/clang++ compilers.
COPY scripts/build-clang.sh /tmp/
ENV LLVM_BUILD_TARGETS=X86
RUN ./build-clang.sh
ENV CC=clang CXX=clang++

# Build zstd to enable `llvm.libzstd`.
COPY scripts/build-zstd.sh /tmp/
RUN ./build-zstd.sh

COPY scripts/sccache.sh /scripts/
RUN sh /scripts/sccache.sh

ENV PGO_HOST=x86_64-unknown-linux-gnu

ENV HOSTS=x86_64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS \
      --enable-full-tools \
      --enable-sanitizers \
      --enable-profiler \
      --enable-compiler-docs \
      --set target.x86_64-unknown-linux-gnu.linker=clang \
      --set target.x86_64-unknown-linux-gnu.ar=/rustroot/bin/llvm-ar \
      --set target.x86_64-unknown-linux-gnu.ranlib=/rustroot/bin/llvm-ranlib \
      --set llvm.thin-lto=true \
      --set llvm.ninja=false \
      --set llvm.libzstd=true \
      --set rust.jemalloc \
      --set rust.use-lld=true \
      --set rust.lto=thin \
      --set rust.codegen-units=1

ARG SCRIPT_ARG

COPY host-x86_64/dist-x86_64-linux/dist.sh /scripts/
COPY host-x86_64/dist-x86_64-linux/dist-alt.sh /scripts/

ENV SCRIPT /scripts/${SCRIPT_ARG}

ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=clang

# This is the only builder which will create source tarballs
ENV DIST_SRC 1

# When we build cargo in this container, we don't want it to use the system
# libcurl, instead it should compile its own.
ENV LIBCURL_NO_PKG_CONFIG 1

ENV DIST_REQUIRE_ALL_TOOLS 1

# FIXME: Without this, LLVMgold.so incorrectly resolves to the system
# libstdc++, instead of the one we build.
ENV LD_PRELOAD=/rustroot/lib64/libstdc++.so.6
