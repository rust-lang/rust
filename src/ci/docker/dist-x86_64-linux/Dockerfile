FROM centos:6

RUN yum -y update \
   && yum -y install centos-release-scl epel-release \
   && yum -y install \
      curl \
      bzip2 \
      cmake3 \
      devtoolset-6-binutils \
      devtoolset-6-gcc \
      devtoolset-6-gcc-c++ \
      devtoolset-6-make \
      git \
      glibc-devel \
      openssl-devel \
      perl \
      python27-python \
      zlib-devel \
      file \
      xz \
      which \
      pkgconfig \
      wget \
      autoconf \
      gettext \
    && yum clean all

# Unfortunately there's no wonderful packaging for cmake3
RUN ln -s /usr/bin/cmake3 /usr/local/bin/cmake

RUN curl -Lo /usr/local/bin/dumb-init \
      https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64 && \
      chmod +x /usr/local/bin/dumb-init
ENTRYPOINT ["/usr/local/bin/dumb-init", "--", "scl", "enable", "python27", "devtoolset-6"]

RUN curl -o /usr/local/bin/sccache \
      https://s3.amazonaws.com/rust-lang-ci/rust-ci-mirror/2017-05-12-sccache-x86_64-unknown-linux-musl && \
      chmod +x /usr/local/bin/sccache

ENV HOSTS=x86_64-unknown-linux-gnu

ENV RUST_CONFIGURE_ARGS \
      --host=$HOSTS \
      --enable-extended \
      --enable-sanitizers
ENV SCRIPT python2.7 ../x.py dist --host $HOSTS --target $HOSTS

# This is the only builder which will create source tarballs
ENV DIST_SRC 1

# When we build cargo in this container, we don't want it to use the system
# libcurl, instead it should compile its own.
ENV LIBCURL_NO_PKG_CONFIG 1
