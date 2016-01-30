# mips-unknown-linux-musl configuration
CC_mips-unknown-linux-musl=mips-linux-musl-gcc
CXX_mips-unknown-linux-musl=mips-linux-musl-g++
CPP_mips-unknown-linux-musl=mips-linux-musl-gcc -E
AR_mips-unknown-linux-musl=mips-linux-musl-ar
CFG_LIB_NAME_mips-unknown-linux-musl=lib$(1).so
CFG_STATIC_LIB_NAME_mips-unknown-linux-musl=lib$(1).a
CFG_LIB_GLOB_mips-unknown-linux-musl=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_mips-unknown-linux-musl=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_mips-unknown-linux-musl := -mips32r2 -msoft-float -mabi=32 $(CFLAGS)
CFG_GCCISH_CFLAGS_mips-unknown-linux-musl := -Wall -g -fPIC -mips32r2 -msoft-float -mabi=32 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_mips-unknown-linux-musl := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_mips-unknown-linux-musl := -shared -fPIC -g -mips32r2 -msoft-float -mabi=32
CFG_GCCISH_DEF_FLAG_mips-unknown-linux-musl := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_mips-unknown-linux-musl :=
CFG_INSTALL_NAME_mips-unknown-linux-musl =
CFG_EXE_SUFFIX_mips-unknown-linux-musl =
CFG_WINDOWSY_mips-unknown-linux-musl :=
CFG_UNIXY_mips-unknown-linux-musl := 1
CFG_LDPATH_mips-unknown-linux-musl :=
CFG_RUN_mips-unknown-linux-musl=
CFG_RUN_TARG_mips-unknown-linux-musl=
RUSTC_FLAGS_mips-unknown-linux-musl :=
CFG_GNU_TRIPLE_mips-unknown-linux-musl := mips-unknown-linux-musl
