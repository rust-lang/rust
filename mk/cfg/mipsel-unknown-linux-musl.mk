# mipsel-unknown-linux-musl configuration
CC_mipsel-unknown-linux-musl=mipsel-linux-musl-gcc
CXX_mipsel-unknown-linux-musl=mipsel-linux-musl-g++
CPP_mipsel-unknown-linux-musl=mipsel-linux-musl-gcc
AR_mipsel-unknown-linux-musl=mipsel-linux-musl-ar
CFG_LIB_NAME_mipsel-unknown-linux-musl=lib$(1).so
CFG_STATIC_LIB_NAME_mipsel-unknown-linux-musl=lib$(1).a
CFG_LIB_GLOB_mipsel-unknown-linux-musl=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_mipsel-unknown-linux-musl=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_mipsel-unknown-linux-musl := -mips32 -mabi=32 $(CFLAGS)
CFG_GCCISH_CFLAGS_mipsel-unknown-linux-musl := -Wall -g -fPIC -mips32 -mabi=32 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_mipsel-unknown-linux-musl := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_mipsel-unknown-linux-musl := -shared -fPIC -g -mips32
CFG_GCCISH_DEF_FLAG_mipsel-unknown-linux-musl := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_mipsel-unknown-linux-musl :=
CFG_INSTALL_NAME_mipsel-unknown-linux-musl =
CFG_EXE_SUFFIX_mipsel-unknown-linux-musl :=
CFG_WINDOWSY_mipsel-unknown-linux-musl :=
CFG_UNIXY_mipsel-unknown-linux-musl := 1
CFG_LDPATH_mipsel-unknown-linux-musl :=
CFG_RUN_mipsel-unknown-linux-musl=
CFG_RUN_TARG_mipsel-unknown-linux-musl=
RUSTC_FLAGS_mipsel-unknown-linux-musl :=
CFG_GNU_TRIPLE_mipsel-unknown-linux-musl := mipsel-unknown-linux-musl
