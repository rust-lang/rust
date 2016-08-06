# i686-unknown-linux-gnu configuration
CC_i686-unknown-linux-gnu=$(CC)
CXX_i686-unknown-linux-gnu=$(CXX)
CPP_i686-unknown-linux-gnu=$(CPP)
AR_i686-unknown-linux-gnu=$(AR)
CFG_LIB_NAME_i686-unknown-linux-gnu=lib$(1).so
CFG_STATIC_LIB_NAME_i686-unknown-linux-gnu=lib$(1).a
CFG_LIB_GLOB_i686-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_i686-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_i686-unknown-linux-gnu := -m32 $(CFLAGS)
CFG_GCCISH_CFLAGS_i686-unknown-linux-gnu :=  -g -fPIC -m32 $(CFLAGS) -march=i686
CFG_GCCISH_CXXFLAGS_i686-unknown-linux-gnu := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_i686-unknown-linux-gnu := -shared -fPIC -ldl -pthread  -lrt -g -m32
CFG_GCCISH_DEF_FLAG_i686-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_i686-unknown-linux-gnu :=
CFG_INSTALL_NAME_i686-unknown-linux-gnu =
CFG_EXE_SUFFIX_i686-unknown-linux-gnu =
CFG_WINDOWSY_i686-unknown-linux-gnu :=
CFG_UNIXY_i686-unknown-linux-gnu := 1
CFG_LDPATH_i686-unknown-linux-gnu :=
CFG_RUN_i686-unknown-linux-gnu=$(2)
CFG_RUN_TARG_i686-unknown-linux-gnu=$(call CFG_RUN_i686-unknown-linux-gnu,,$(2))
CFG_GNU_TRIPLE_i686-unknown-linux-gnu := i686-unknown-linux-gnu
