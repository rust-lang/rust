# x86_64-pc-openbsd-elf configuration
CC_x86_64-unknown-openbsd=$(CC)
CXX_x86_64-unknown-openbsd=$(CXX)
CPP_x86_64-unknown-openbsd=$(CPP)
AR_x86_64-unknown-openbsd=$(AR)
CFG_LIB_NAME_x86_64-unknown-openbsd=lib$(1).so
CFG_STATIC_LIB_NAME_x86_64-unknown-openbsd=lib$(1).a
CFG_LIB_GLOB_x86_64-unknown-openbsd=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_x86_64-unknown-openbsd=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_x86_64-unknown-openbsd := -m64 -I/usr/include $(CFLAGS)
CFG_GCCISH_CFLAGS_x86_64-unknown-openbsd := -Wall -Werror -g -fPIC -m64 -I/usr/include $(CFLAGS)
CFG_GCCISH_LINK_FLAGS_x86_64-unknown-openbsd := -shared -fPIC -g -pthread -m64
CFG_GCCISH_DEF_FLAG_x86_64-unknown-openbsd := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_x86_64-unknown-openbsd := -Wl,-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_x86_64-unknown-openbsd := -Wl,-no-whole-archive
CFG_DEF_SUFFIX_x86_64-unknown-openbsd := .bsd.def
CFG_LLC_FLAGS_x86_64-unknown-openbsd :=
CFG_INSTALL_NAME_x86_64-unknown-openbsd =
CFG_EXE_SUFFIX_x86_64-unknown-openbsd :=
CFG_WINDOWSY_x86_64-unknown-openbsd :=
CFG_UNIXY_x86_64-unknown-openbsd := 1
CFG_PATH_MUNGE_x86_64-unknown-openbsd :=
CFG_LDPATH_x86_64-unknown-openbsd :=
CFG_RUN_x86_64-unknown-openbsd=$(2)
CFG_RUN_TARG_x86_64-unknown-openbsd=$(call CFG_RUN_x86_64-unknown-openbsd,,$(2))
CFG_GNU_TRIPLE_x86_64-unknown-openbsd := x86_64-unknown-openbsd
