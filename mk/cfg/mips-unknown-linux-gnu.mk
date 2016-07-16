# mips-unknown-linux-gnu configuration
CC_mips-unknown-linux-gnu=mips-linux-gnu-gcc
CXX_mips-unknown-linux-gnu=mips-linux-gnu-g++
CPP_mips-unknown-linux-gnu=mips-linux-gnu-gcc -E
AR_mips-unknown-linux-gnu=mips-linux-gnu-ar
CFG_LIB_NAME_mips-unknown-linux-gnu=lib$(1).so
CFG_STATIC_LIB_NAME_mips-unknown-linux-gnu=lib$(1).a
CFG_LIB_GLOB_mips-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_mips-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_mips-unknown-linux-gnu := -mips32r2 -mabi=32 $(CFLAGS)
CFG_GCCISH_CFLAGS_mips-unknown-linux-gnu := -Wall -g -fPIC -mips32r2 -mabi=32 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_mips-unknown-linux-gnu := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_mips-unknown-linux-gnu := -shared -fPIC -g -mips32r2 -mabi=32
CFG_GCCISH_DEF_FLAG_mips-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_mips-unknown-linux-gnu :=
CFG_INSTALL_NAME_mips-unknown-linux-gnu =
CFG_EXE_SUFFIX_mips-unknown-linux-gnu :=
CFG_WINDOWSY_mips-unknown-linux-gnu :=
CFG_UNIXY_mips-unknown-linux-gnu := 1
CFG_LDPATH_mips-unknown-linux-gnu :=
CFG_RUN_mips-unknown-linux-gnu=
CFG_RUN_TARG_mips-unknown-linux-gnu=
RUSTC_FLAGS_mips-unknown-linux-gnu :=
CFG_GNU_TRIPLE_mips-unknown-linux-gnu := mips-unknown-linux-gnu
