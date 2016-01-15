# powerpc64le-unknown-linux-gnu configuration
CROSS_PREFIX_powerpc64le-unknown-linux-gnu=powerpc64le-linux-gnu-
CC_powerpc64le-unknown-linux-gnu=$(CC)
CXX_powerpc64le-unknown-linux-gnu=$(CXX)
CPP_powerpc64le-unknown-linux-gnu=$(CPP)
AR_powerpc64le-unknown-linux-gnu=$(AR)
CFG_LIB_NAME_powerpc64le-unknown-linux-gnu=lib$(1).so
CFG_STATIC_LIB_NAME_powerpc64le-unknown-linux-gnu=lib$(1).a
CFG_LIB_GLOB_powerpc64le-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_powerpc64le-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_CFLAGS_powerpc64le-unknown-linux-gnu := -m64 $(CFLAGS)
CFG_GCCISH_CFLAGS_powerpc64le-unknown-linux-gnu := -Wall -Werror -g -fPIC -m64 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_powerpc64le-unknown-linux-gnu := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_powerpc64le-unknown-linux-gnu := -shared -fPIC -ldl -pthread  -lrt -g -m64
CFG_GCCISH_DEF_FLAG_powerpc64le-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_powerpc64le-unknown-linux-gnu :=
CFG_INSTALL_NAME_powerpc64le-unknown-linux-gnu =
CFG_EXE_SUFFIX_powerpc64le-unknown-linux-gnu =
CFG_WINDOWSY_powerpc64le-unknown-linux-gnu :=
CFG_UNIXY_powerpc64le-unknown-linux-gnu := 1
CFG_LDPATH_powerpc64le-unknown-linux-gnu :=
CFG_RUN_powerpc64le-unknown-linux-gnu=$(2)
CFG_RUN_TARG_powerpc64le-unknown-linux-gnu=$(call CFG_RUN_powerpc64le-unknown-linux-gnu,,$(2))
CFG_GNU_TRIPLE_powerpc64le-unknown-linux-gnu := powerpc64le-unknown-linux-gnu
