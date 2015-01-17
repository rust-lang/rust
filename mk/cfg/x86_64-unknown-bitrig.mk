# x86_64-unknown-bitrig-elf configuration
CC_x86_64-unknown-bitrig=$(CC)
CXX_x86_64-unknown-bitrig=$(CXX)
CPP_x86_64-unknown-bitrig=$(CPP)
AR_x86_64-unknown-bitrig=$(AR)
CFG_LIB_NAME_x86_64-unknown-bitrig=lib$(1).so
CFG_STATIC_LIB_NAME_x86_64-unknown-bitrig=lib$(1).a
CFG_LIB_GLOB_x86_64-unknown-bitrig=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_x86_64-unknown-bitrig=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_x86_64-unknown-bitrig := -m64 -I/usr/include $(CFLAGS)
CFG_GCCISH_CFLAGS_x86_64-unknown-bitrig := -Wall -Werror -fPIC -m64 -I/usr/include $(CFLAGS)
CFG_GCCISH_LINK_FLAGS_x86_64-unknown-bitrig := -shared -pic -pthread -m64 $(LDFLAGS)
CFG_GCCISH_DEF_FLAG_x86_64-unknown-bitrig := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_x86_64-unknown-bitrig := -Wl,-pic -Wl,-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_x86_64-unknown-bitrig := -Wl,-no-whole-archive
CFG_DEF_SUFFIX_x86_64-unknown-bitrig := .bsd.def
CFG_LLC_FLAGS_x86_64-unknown-bitrig :=
CFG_INSTALL_NAME_x86_64-unknown-bitrig =
CFG_EXE_SUFFIX_x86_64-unknown-bitrig :=
CFG_WINDOWSY_x86_64-unknown-bitrig :=
CFG_UNIXY_x86_64-unknown-bitrig := 1
CFG_PATH_MUNGE_x86_64-unknown-bitrig :=
CFG_LDPATH_x86_64-unknown-bitrig :=
CFG_RUN_x86_64-unknown-bitrig=$(2)
CFG_RUN_TARG_x86_64-unknown-bitrig=$(call CFG_RUN_x86_64-unknown-bitrig,,$(2))
CFG_GNU_TRIPLE_x86_64-unknown-bitrig := x86_64-unknown-bitrig
