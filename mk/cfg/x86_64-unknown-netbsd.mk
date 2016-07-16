# x86_64-unknown-netbsd configuration
CROSS_PREFIX_x86_64-unknown-netbsd=x86_64-unknown-netbsd-
CC_x86_64-unknown-netbsd=$(CC)
CXX_x86_64-unknown-netbsd=$(CXX)
CPP_x86_64-unknown-netbsd=$(CPP)
AR_x86_64-unknown-netbsd=$(AR)
CFG_LIB_NAME_x86_64-unknown-netbsd=lib$(1).so
CFG_STATIC_LIB_NAME_x86_64-unknown-netbsd=lib$(1).a
CFG_LIB_GLOB_x86_64-unknown-netbsd=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_x86_64-unknown-netbsd=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_x86_64-unknown-netbsd := -I/usr/local/include $(CFLAGS)
CFG_GCCISH_CFLAGS_x86_64-unknown-netbsd :=  -g -fPIC -I/usr/local/include $(CFLAGS)
CFG_GCCISH_LINK_FLAGS_x86_64-unknown-netbsd := -shared -fPIC -g -pthread  -lrt
CFG_GCCISH_DEF_FLAG_x86_64-unknown-netbsd := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_x86_64-unknown-netbsd :=
CFG_INSTALL_NAME_x86_64-unknown-netbsd =
CFG_EXE_SUFFIX_x86_64-unknown-netbsd :=
CFG_WINDOWSY_x86_64-unknown-netbsd :=
CFG_UNIXY_x86_64-unknown-netbsd := 1
CFG_LDPATH_x86_64-unknown-netbsd :=
CFG_RUN_x86_64-unknown-netbsd=$(2)
CFG_RUN_TARG_x86_64-unknown-netbsd=$(call CFG_RUN_x86_64-unknown-netbsd,,$(2))
CFG_GNU_TRIPLE_x86_64-unknown-netbsd := x86_64-unknown-netbsd
