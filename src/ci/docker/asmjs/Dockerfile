FROM rust-ci-base

# emscripten
COPY scripts/emscripten.sh /scripts/
RUN bash /scripts/emscripten.sh

# env
ENV PATH=$PATH:/emsdk-portable
ENV PATH=$PATH:/emsdk-portable/clang/e1.37.13_64bit/
ENV PATH=$PATH:/emsdk-portable/emscripten/1.37.13/
ENV PATH=$PATH:/emsdk-portable/node/4.1.1_64bit/bin/
ENV EMSCRIPTEN=/emsdk-portable/emscripten/1.37.13/
ENV BINARYEN_ROOT=/emsdk-portable/clang/e1.37.13_64bit/binaryen/
ENV EM_CONFIG=/emsdk-portable/.emscripten

ENV TARGETS=asmjs-unknown-emscripten

ENV RUST_CONFIGURE_ARGS --target=$TARGETS

ENV SCRIPT python2.7 ../x.py test --target $TARGETS
