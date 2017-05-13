# s390x-unknown-linux-gnu configuration
CROSS_PREFIX_s390x-unknown-linux-gnu=s390x-linux-gnu-
CC_s390x-unknown-linux-gnu=$(CC)
CXX_s390x-unknown-linux-gnu=$(CXX)
CPP_s390x-unknown-linux-gnu=$(CPP)
AR_s390x-unknown-linux-gnu=$(AR)
CFG_LIB_NAME_s390x-unknown-linux-gnu=lib$(1).so
CFG_STATIC_LIB_NAME_s390x-unknown-linux-gnu=lib$(1).a
CFG_LIB_GLOB_s390x-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_s390x-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_CFLAGS_s390x-unknown-linux-gnu := -m64 $(CFLAGS)
CFG_GCCISH_CFLAGS_s390x-unknown-linux-gnu :=  -g -fPIC -m64 $(CFLAGS)
CFG_GCCISH_CXXFLAGS_s390x-unknown-linux-gnu := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_s390x-unknown-linux-gnu := -shared -fPIC -ldl -pthread  -lrt -g -m64
CFG_GCCISH_DEF_FLAG_s390x-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_s390x-unknown-linux-gnu :=
CFG_INSTALL_NAME_s390x-unknown-linux-gnu =
CFG_EXE_SUFFIX_s390x-unknown-linux-gnu =
CFG_WINDOWSY_s390x-unknown-linux-gnu :=
CFG_UNIXY_s390x-unknown-linux-gnu := 1
CFG_LDPATH_s390x-unknown-linux-gnu :=
CFG_RUN_s390x-unknown-linux-gnu=$(2)
CFG_RUN_TARG_s390x-unknown-linux-gnu=$(call CFG_RUN_s390x-unknown-linux-gnu,,$(2))
CFG_GNU_TRIPLE_s390x-unknown-linux-gnu := s390x-unknown-linux-gnu
