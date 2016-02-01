# mipsel-unknown-linux-gnu configuration
CC_mipsel-unknown-linux-gnu=mipsel-linux-gnu-gcc
CXX_mipsel-unknown-linux-gnu=mipsel-linux-gnu-g++
CPP_mipsel-unknown-linux-gnu=mipsel-linux-gnu-gcc
AR_mipsel-unknown-linux-gnu=mipsel-linux-gnu-ar
CFG_LIB_NAME_mipsel-unknown-linux-gnu=lib$(1).so
CFG_STATIC_LIB_NAME_mipsel-unknown-linux-gnu=lib$(1).a
CFG_LIB_GLOB_mipsel-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_mipsel-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_mipsel-unknown-linux-gnu := -mips32 -mabi=32 $(CFLAGS)
CFG_GCCISH_CFLAGS_mipsel-unknown-linux-gnu := -Wall -g -fPIC -mips32 -mabi=32 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_mipsel-unknown-linux-gnu := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_mipsel-unknown-linux-gnu := -shared -fPIC -g -mips32
CFG_GCCISH_DEF_FLAG_mipsel-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_mipsel-unknown-linux-gnu :=
CFG_INSTALL_NAME_mipsel-unknown-linux-gnu =
CFG_EXE_SUFFIX_mipsel-unknown-linux-gnu :=
CFG_WINDOWSY_mipsel-unknown-linux-gnu :=
CFG_UNIXY_mipsel-unknown-linux-gnu := 1
CFG_LDPATH_mipsel-unknown-linux-gnu :=
CFG_RUN_mipsel-unknown-linux-gnu=
CFG_RUN_TARG_mipsel-unknown-linux-gnu=
RUSTC_FLAGS_mipsel-unknown-linux-gnu :=
CFG_GNU_TRIPLE_mipsel-unknown-linux-gnu := mipsel-unknown-linux-gnu
