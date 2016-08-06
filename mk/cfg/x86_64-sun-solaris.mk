# x86_64-sun-solaris configuration
CROSS_PREFIX_x86_64-sun-solaris=x86_64-sun-solaris2.11-
CC_x86_64-sun-solaris=$(CC)
CXX_x86_64-sun-solaris=$(CXX)
CPP_x86_64-sun-solaris=$(CPP)
AR_x86_64-sun-solaris=$(AR)
CFG_LIB_NAME_x86_64-sun-solaris=lib$(1).so
CFG_STATIC_LIB_NAME_x86_64-sun-solaris=lib$(1).a
CFG_LIB_GLOB_x86_64-sun-solaris=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_x86_64-sun-solaris=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_x86_64-sun-solaris := -I/usr/local/include $(CFLAGS)
CFG_GCCISH_CFLAGS_x86_64-sun-solaris :=  -g -D_POSIX_PTHREAD_SEMANTICS -fPIC -I/usr/local/include $(CFLAGS)
CFG_GCCISH_LINK_FLAGS_x86_64-sun-solaris := -shared -fPIC -g -pthread  -lrt
CFG_GCCISH_DEF_FLAG_x86_64-sun-solaris := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_x86_64-sun-solaris :=
CFG_INSTALL_NAME_x86_64-sun-solaris =
CFG_EXE_SUFFIX_x86_64-sun-solaris :=
CFG_WINDOWSY_x86_64-sun-solaris :=
CFG_UNIXY_x86_64-sun-solaris := 1
CFG_LDPATH_x86_64-sun-solaris :=
CFG_RUN_x86_64-sun-solaris=$(2)
CFG_RUN_TARG_x86_64-sun-solaris=$(call CFG_RUN_x86_64-sun-solaris,,$(2))
CFG_GNU_TRIPLE_x86_64-sun-solaris := x86_64-sun-solaris
