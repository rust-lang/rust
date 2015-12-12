# i686-unknown-freebsd configuration
CC_i686-unknown-freebsd=$(CC)
CXX_i686-unknown-freebsd=$(CXX)
CPP_i686-unknown-freebsd=$(CPP)
AR_i686-unknown-freebsd=$(AR)
CFG_LIB_NAME_i686-unknown-freebsd=lib$(1).so
CFG_STATIC_LIB_NAME_i686-unknown-freebsd=lib$(1).a
CFG_LIB_GLOB_i686-unknown-freebsd=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_i686-unknown-freebsd=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_i686-unknown-freebsd := -m32 -I/usr/local/include $(CFLAGS)
CFG_GCCISH_CFLAGS_i686-unknown-freebsd := -Wall -Werror -g -fPIC -m32 -arch i386 -I/usr/local/include $(CFLAGS)
CFG_GCCISH_LINK_FLAGS_i686-unknown-freebsd := -m32 -shared -fPIC -g -pthread -lrt
CFG_GCCISH_DEF_FLAG_i686-unknown-freebsd := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_i686-unknown-freebsd :=
CFG_INSTALL_NAME_i686-unknown-freebsd =
CFG_EXE_SUFFIX_i686-unknown-freebsd :=
CFG_WINDOWSY_i686-unknown-freebsd :=
CFG_UNIXY_i686-unknown-freebsd := 1
CFG_LDPATH_i686-unknown-freebsd :=
CFG_RUN_i686-unknown-freebsd=$(2)
CFG_RUN_TARG_i686-unknown-freebsd=$(call CFG_RUN_i686-unknown-freebsd,,$(2))
CFG_GNU_TRIPLE_i686-unknown-freebsd := i686-unknown-freebsd
