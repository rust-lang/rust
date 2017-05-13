# i586-unknown-linux-gnu configuration
CC_i586-unknown-linux-gnu=$(CC)
CXX_i586-unknown-linux-gnu=$(CXX)
CPP_i586-unknown-linux-gnu=$(CPP)
AR_i586-unknown-linux-gnu=$(AR)
CFG_LIB_NAME_i586-unknown-linux-gnu=lib$(1).so
CFG_STATIC_LIB_NAME_i586-unknown-linux-gnu=lib$(1).a
CFG_LIB_GLOB_i586-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_i586-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_i586-unknown-linux-gnu := -m32 $(CFLAGS) -march=pentium -Wa,-mrelax-relocations=no
CFG_GCCISH_CFLAGS_i586-unknown-linux-gnu :=  -g -fPIC -m32 $(CFLAGS) -march=pentium -Wa,-mrelax-relocations=no
CFG_GCCISH_CXXFLAGS_i586-unknown-linux-gnu := -fno-rtti $(CXXFLAGS) -march=pentium
CFG_GCCISH_LINK_FLAGS_i586-unknown-linux-gnu := -shared -fPIC -ldl -pthread  -lrt -g -m32
CFG_GCCISH_DEF_FLAG_i586-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_i586-unknown-linux-gnu :=
CFG_INSTALL_NAME_i586-unknown-linux-gnu =
CFG_EXE_SUFFIX_i586-unknown-linux-gnu =
CFG_WINDOWSY_i586-unknown-linux-gnu :=
CFG_UNIXY_i586-unknown-linux-gnu := 1
CFG_LDPATH_i586-unknown-linux-gnu :=
CFG_RUN_i586-unknown-linux-gnu=$(2)
CFG_RUN_TARG_i586-unknown-linux-gnu=$(call CFG_RUN_i586-unknown-linux-gnu,,$(2))
CFG_GNU_TRIPLE_i586-unknown-linux-gnu := i586-unknown-linux-gnu
