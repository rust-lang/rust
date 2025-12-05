# We document platform support for minimum glibc 2.17 and kernel 3.2.
# CentOS 7 has headers for kernel 3.10, but that's fine as long as we don't
# actually use newer APIs in rustc or std without a fallback. It's more
# important that we match glibc for ELF symbol versioning.
FROM centos:7

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
      binutils.i686 \
      glibc-devel.i686 \
      glibc-devel.x86_64 \
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
      xz \
      zlib-devel.i686 \
      zlib-devel.x86_64 \
      && yum clean all

RUN mkdir -p /rustroot/bin

ENV PATH=/rustroot/bin:$PATH
ENV LD_LIBRARY_PATH=/rustroot/lib64:/rustroot/lib32:/rustroot/lib
ENV PKG_CONFIG_PATH=/rustroot/lib/pkgconfig
WORKDIR /tmp
RUN mkdir /home/user
COPY scripts/shared.sh /tmp/

# Need at least GCC 5.1 to compile LLVM nowadays
COPY scripts/build-gcc.sh /tmp/
ENV GCC_VERSION=9.5.0
ENV GCC_BUILD_TARGET=i686-pc-linux-gnu
RUN ./build-gcc.sh && yum remove -y gcc gcc-c++

COPY scripts/cmake.sh /tmp/
RUN ./cmake.sh

# Now build LLVM+Clang, afterwards configuring further compilations to use the
# clang/clang++ compilers.
COPY scripts/build-clang.sh /tmp/
ENV LLVM_BUILD_TARGETS=X86
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
