FROM rust-ci-base

# emscripten
COPY scripts/emscripten.sh /scripts/
RUN bash /scripts/emscripten.sh
COPY wasm32/node.sh /usr/local/bin/node

# env
ENV PATH=$PATH:/emsdk-portable
ENV PATH=$PATH:/emsdk-portable/clang/e1.37.13_64bit/
ENV PATH=$PATH:/emsdk-portable/emscripten/1.37.13/
ENV EMSCRIPTEN=/emsdk-portable/emscripten/1.37.13/
ENV BINARYEN_ROOT=/emsdk-portable/clang/e1.37.13_64bit/binaryen/
ENV EM_CONFIG=/emsdk-portable/.emscripten

ENV TARGETS=wasm32-unknown-emscripten

ENV RUST_CONFIGURE_ARGS --target=$TARGETS

ENV SCRIPT python2.7 ../x.py test --target $TARGETS
