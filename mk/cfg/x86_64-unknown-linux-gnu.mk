# x86_64-unknown-linux-gnu configuration
CC_x86_64-unknown-linux-gnu=$(CC)
CXX_x86_64-unknown-linux-gnu=$(CXX)
CPP_x86_64-unknown-linux-gnu=$(CPP)
AR_x86_64-unknown-linux-gnu=$(AR)
CFG_LIB_NAME_x86_64-unknown-linux-gnu=lib$(1).so
CFG_STATIC_LIB_NAME_x86_64-unknown-linux-gnu=lib$(1).a
CFG_LIB_GLOB_x86_64-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_x86_64-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_x86_64-unknown-linux-gnu := -m64
CFG_GCCISH_CFLAGS_x86_64-unknown-linux-gnu :=  -g -fPIC -m64
CFG_GCCISH_CXXFLAGS_x86_64-unknown-linux-gnu := -fno-rtti
CFG_GCCISH_LINK_FLAGS_x86_64-unknown-linux-gnu := -shared -fPIC -ldl -pthread  -lrt -g -m64
CFG_GCCISH_DEF_FLAG_x86_64-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_x86_64-unknown-linux-gnu :=
CFG_INSTALL_NAME_x86_64-unknown-linux-gnu =
CFG_EXE_SUFFIX_x86_64-unknown-linux-gnu =
CFG_WINDOWSY_x86_64-unknown-linux-gnu :=
CFG_UNIXY_x86_64-unknown-linux-gnu := 1
CFG_LDPATH_x86_64-unknown-linux-gnu :=
CFG_RUN_x86_64-unknown-linux-gnu=$(2)
CFG_RUN_TARG_x86_64-unknown-linux-gnu=$(call CFG_RUN_x86_64-unknown-linux-gnu,,$(2))
CFG_GNU_TRIPLE_x86_64-unknown-linux-gnu := x86_64-unknown-linux-gnu
