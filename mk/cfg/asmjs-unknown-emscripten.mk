# asmjs-unknown-emscripten configuration
CC_asmjs-unknown-emscripten=emcc
CXX_asmjs-unknown-emscripten=em++
CPP_asmjs-unknown-emscripten=$(CPP)
AR_asmjs-unknown-emscripten=emar
CFG_LIB_NAME_asmjs-unknown-emscripten=lib$(1).so
CFG_STATIC_LIB_NAME_asmjs-unknown-emscripten=lib$(1).a
CFG_LIB_GLOB_asmjs-unknown-emscripten=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_asmjs-unknown-emscripten=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_asmjs-unknown-emscripten := -m32 $(CFLAGS)
# NB: The EMSCRIPTEN environment variable is set by the emscripten SDK.
# This is only needed here to so the compiler-rt build can find unwind.h,
# but the asmjs target *doesn't even link to compiler-rt*.
CFG_GCCISH_CFLAGS_asmjs-unknown-emscripten := -Wall -Werror -g -fPIC -m32 $(CFLAGS) \
    -I$(EMSCRIPTEN)/system/lib/libcxxabi/include
CFG_GCCISH_CXXFLAGS_asmjs-unknown-emscripten := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_asmjs-unknown-emscripten := -shared -fPIC -ldl -pthread  -lrt -g -m32
CFG_GCCISH_DEF_FLAG_asmjs-unknown-emscripten := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_asmjs-unknown-emscripten :=
CFG_INSTALL_NAME_asmjs-unknown-emscripten =
CFG_EXE_SUFFIX_asmjs-unknown-emscripten =
CFG_WINDOWSY_asmjs-unknown-emscripten :=
CFG_UNIXY_asmjs-unknown-emscripten := 1
CFG_LDPATH_asmjs-unknown-emscripten :=
CFG_RUN_asmjs-unknown-emscripten=$(2)
CFG_RUN_TARG_asmjs-unknown-emscripten=$(call CFG_RUN_asmjs-unknown-emscripten,,$(2))
CFG_GNU_TRIPLE_asmjs-unknown-emscripten := asmjs-unknown-emscripten
