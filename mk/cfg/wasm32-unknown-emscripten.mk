# wasm32-unknown-emscripten configuration
CC_wasm32-unknown-emscripten=emcc
CXX_wasm32-unknown-emscripten=em++
CPP_wasm32-unknown-emscripten=$(CPP)
AR_wasm32-unknown-emscripten=emar
CFG_LIB_NAME_wasm32-unknown-emscripten=lib$(1).so
CFG_STATIC_LIB_NAME_wasm32-unknown-emscripten=lib$(1).a
CFG_LIB_GLOB_wasm32-unknown-emscripten=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_wasm32-unknown-emscripten=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_wasm32-unknown-emscripten := -m32 $(CFLAGS)
CFG_GCCISH_CFLAGS_wasm32-unknown-emscripten :=  -g -fPIC -m32 -s BINARYEN=1 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_wasm32-unknown-emscripten := -fno-rtti -s BINARYEN=1 $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_wasm32-unknown-emscripten := -shared -fPIC -ldl -pthread  -lrt -g -m32 -s BINARYEN=1
CFG_GCCISH_DEF_FLAG_wasm32-unknown-emscripten := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_wasm32-unknown-emscripten :=
CFG_INSTALL_NAME_wasm32-unknown-emscripten =
CFG_EXE_SUFFIX_wasm32-unknown-emscripten =
CFG_WINDOWSY_wasm32-unknown-emscripten :=
CFG_UNIXY_wasm32-unknown-emscripten := 1
CFG_LDPATH_wasm32-unknown-emscripten :=
CFG_RUN_wasm32-unknown-emscripten=$(2)
CFG_RUN_TARG_wasm32-unknown-emscripten=$(call CFG_RUN_wasm32-unknown-emscripten,,$(2))
CFG_GNU_TRIPLE_wasm32-unknown-emscripten := wasm32-unknown-emscripten
CFG_DISABLE_JEMALLOC_wasm32-unknown-emscripten := 1
