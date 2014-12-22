# mipsel-unknown-linux-gnu configuration
CC_mipsel-unknown-linux-gnu=mipsel-unknown-linux-gnu-gcc
CXX_mipsel-unknown-linux-gnu=mipsel-unknown-linux-gnu-g++
CPP_mipsel-unknown-linux-gnu=mipsel-unknown-linux-gnu-gcc
AR_mipsel-unknown-linux-gnu=mipsel-unknown-linux-gnu-ar
CFG_LIB_NAME_mipsel-unknown-linux-gnu=lib$(1).so
CFG_STATIC_LIB_NAME_mipsel-unknown-linux-gnu=lib$(1).a
CFG_LIB_GLOB_mipsel-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_mipsel-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_mipsel-unknown-linux-gnu := -mips32 -mabi=32 $(CFLAGS)
CFG_GCCISH_CFLAGS_mipsel-unknown-linux-gnu := -Wall -g -fPIC -mips32 -mabi=32 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_mipsel-unknown-linux-gnu := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_mipsel-unknown-linux-gnu := -shared -fPIC -g -mips32
CFG_GCCISH_DEF_FLAG_mipsel-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_mipsel-unknown-linux-gnu := -Wl,-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_mipsel-unknown-linux-gnu := -Wl,-no-whole-archive
CFG_DEF_SUFFIX_mipsel-unknown-linux-gnu := .linux.def
CFG_LLC_FLAGS_mipsel-unknown-linux-gnu :=
CFG_INSTALL_NAME_mipsel-unknown-linux-gnu =
CFG_EXE_SUFFIX_mipsel-unknown-linux-gnu :=
CFG_WINDOWSY_mipsel-unknown-linux-gnu :=
CFG_UNIXY_mipsel-unknown-linux-gnu := 1
CFG_PATH_MUNGE_mipsel-unknown-linux-gnu := true
CFG_LDPATH_mipsel-unknown-linux-gnu :=
CFG_RUN_mipsel-unknown-linux-gnu=
CFG_RUN_TARG_mipsel-unknown-linux-gnu=
RUSTC_FLAGS_mipsel-unknown-linux-gnu := -C target-cpu=mips32 -C target-feature="+mips32,+o32"
CFG_GNU_TRIPLE_mipsel-unknown-linux-gnu := mipsel-unknown-linux-gnu
