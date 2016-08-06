# x86_64-unknown-freebsd configuration
CC_x86_64-unknown-freebsd=$(CC)
CXX_x86_64-unknown-freebsd=$(CXX)
CPP_x86_64-unknown-freebsd=$(CPP)
AR_x86_64-unknown-freebsd=$(AR)
CFG_LIB_NAME_x86_64-unknown-freebsd=lib$(1).so
CFG_STATIC_LIB_NAME_x86_64-unknown-freebsd=lib$(1).a
CFG_LIB_GLOB_x86_64-unknown-freebsd=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_x86_64-unknown-freebsd=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_x86_64-unknown-freebsd := -I/usr/local/include $(CFLAGS)
CFG_GCCISH_CFLAGS_x86_64-unknown-freebsd :=  -g -fPIC -I/usr/local/include $(CFLAGS)
CFG_GCCISH_LINK_FLAGS_x86_64-unknown-freebsd := -shared -fPIC -g -pthread  -lrt
CFG_GCCISH_DEF_FLAG_x86_64-unknown-freebsd := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_x86_64-unknown-freebsd :=
CFG_INSTALL_NAME_x86_64-unknown-freebsd =
CFG_EXE_SUFFIX_x86_64-unknown-freebsd :=
CFG_WINDOWSY_x86_64-unknown-freebsd :=
CFG_UNIXY_x86_64-unknown-freebsd := 1
CFG_LDPATH_x86_64-unknown-freebsd :=
CFG_RUN_x86_64-unknown-freebsd=$(2)
CFG_RUN_TARG_x86_64-unknown-freebsd=$(call CFG_RUN_x86_64-unknown-freebsd,,$(2))
CFG_GNU_TRIPLE_x86_64-unknown-freebsd := x86_64-unknown-freebsd
