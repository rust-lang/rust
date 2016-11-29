# i686-unknown-openbsd configuration
CC_i686-unknown-openbsd=$(CC)
CXX_i686-unknown-openbsd=$(CXX)
CPP_i686-unknown-openbsd=$(CPP)
AR_i686-unknown-openbsd=$(AR)
CFG_LIB_NAME_i686-unknown-openbsd=lib$(1).so
CFG_STATIC_LIB_NAME_i686-unknown-openbsd=lib$(1).a
CFG_LIB_GLOB_i686-unknown-openbsd=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_i686-unknown-openbsd=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_i686-unknown-openbsd := -m32 -I/usr/include $(CFLAGS)
CFG_GCCISH_CFLAGS_i686-unknown-openbsd :=  -g -fPIC -m32 -I/usr/include $(CFLAGS)
CFG_GCCISH_LINK_FLAGS_i686-unknown-openbsd := -shared -fPIC -g -pthread -m32
CFG_GCCISH_DEF_FLAG_i686-unknown-openbsd := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_i686-unknown-openbsd :=
CFG_INSTALL_NAME_i686-unknown-openbsd =
CFG_EXE_SUFFIX_i686-unknown-openbsd :=
CFG_WINDOWSY_i686-unknown-openbsd :=
CFG_UNIXY_i686-unknown-openbsd := 1
CFG_LDPATH_i686-unknown-openbsd :=
CFG_RUN_i686-unknown-openbsd=$(2)
CFG_RUN_TARG_i686-unknown-openbsd=$(call CFG_RUN_i686-unknown-openbsd,,$(2))
CFG_GNU_TRIPLE_i686-unknown-openbsd := i686-unknown-openbsd
RUSTC_FLAGS_i686-unknown-openbsd=-C linker=$(call FIND_COMPILER,$(CC))
CFG_DISABLE_JEMALLOC_i686-unknown-openbsd := 1
