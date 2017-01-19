FROM ubuntu:16.04

RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
  g++ \
  make \
  file \
  curl \
  ca-certificates \
  python2.7 \
  git \
  cmake \
  unzip \
  expect \
  openjdk-9-jre \
  sudo \
  libstdc++6:i386 \
  xz-utils

WORKDIR /android/
ENV PATH=$PATH:/android/ndk-arm-9/bin:/android/sdk/tools:/android/sdk/platform-tools

COPY install-ndk.sh install-sdk.sh accept-licenses.sh /android/
RUN sh /android/install-ndk.sh
RUN sh /android/install-sdk.sh

RUN curl -OL https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64.deb && \
    dpkg -i dumb-init_*.deb && \
    rm dumb-init_*.deb

COPY start-emulator.sh /android/

ENTRYPOINT ["/usr/bin/dumb-init", "--", "/android/start-emulator.sh"]

ENV SCCACHE_DIGEST=7237e38e029342fa27b7ac25412cb9d52554008b12389727320bd533fd7f05b6a96d55485f305caf95e5c8f5f97c3313e10012ccad3e752aba2518f3522ba783
RUN curl -L https://api.pub.build.mozilla.org/tooltool/sha512/$SCCACHE_DIGEST | \
      tar xJf - -C /usr/local/bin --strip-components=1

ENV TARGETS=arm-linux-androideabi
ENV TARGETS=$TARGETS,i686-linux-android
ENV TARGETS=$TARGETS,aarch64-linux-android
ENV TARGETS=$TARGETS,armv7-linux-androideabi

ENV RUST_CONFIGURE_ARGS \
      --target=$TARGETS \
      --arm-linux-androideabi-ndk=/android/ndk-arm-9 \
      --armv7-linux-androideabi-ndk=/android/ndk-arm-9 \
      --i686-linux-android-ndk=/android/ndk-x86-9 \
      --aarch64-linux-android-ndk=/android/ndk-aarch64

# Just a smoke test in dist to see if this works for now, we should expand this
# to all the targets above eventually.
ENV SCRIPT \
  python2.7 ../x.py test --target arm-linux-androideabi && \
  python2.7 ../x.py dist --target $TARGETS
