# powerpc-unknown-linux-gnu configuration
CROSS_PREFIX_powerpc-unknown-linux-gnu=powerpc-linux-gnu-
CC_powerpc-unknown-linux-gnu=$(CC)
CXX_powerpc-unknown-linux-gnu=$(CXX)
CPP_powerpc-unknown-linux-gnu=$(CPP)
AR_powerpc-unknown-linux-gnu=$(AR)
CFG_LIB_NAME_powerpc-unknown-linux-gnu=lib$(1).so
CFG_STATIC_LIB_NAME_powerpc-unknown-linux-gnu=lib$(1).a
CFG_LIB_GLOB_powerpc-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_powerpc-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_CFLAGS_powerpc-unknown-linux-gnu := -m32 $(CFLAGS)
CFG_GCCISH_CFLAGS_powerpc-unknown-linux-gnu :=  -g -fPIC -m32 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_powerpc-unknown-linux-gnu := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_powerpc-unknown-linux-gnu := -shared -fPIC -ldl -pthread  -lrt -g -m32
CFG_GCCISH_DEF_FLAG_powerpc-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_powerpc-unknown-linux-gnu :=
CFG_INSTALL_NAME_powerpc-unknown-linux-gnu =
CFG_EXE_SUFFIX_powerpc-unknown-linux-gnu =
CFG_WINDOWSY_powerpc-unknown-linux-gnu :=
CFG_UNIXY_powerpc-unknown-linux-gnu := 1
CFG_LDPATH_powerpc-unknown-linux-gnu :=
CFG_RUN_powerpc-unknown-linux-gnu=$(2)
CFG_RUN_TARG_powerpc-unknown-linux-gnu=$(call CFG_RUN_powerpc-unknown-linux-gnu,,$(2))
CFG_GNU_TRIPLE_powerpc-unknown-linux-gnu := powerpc-unknown-linux-gnu
