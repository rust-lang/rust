# x86_64-pc-dragonfly-elf configuration
CC_x86_64-unknown-dragonfly=$(CC)
CXX_x86_64-unknown-dragonfly=$(CXX)
CPP_x86_64-unknown-dragonfly=$(CPP)
AR_x86_64-unknown-dragonfly=$(AR)
CFG_LIB_NAME_x86_64-unknown-dragonfly=lib$(1).so
CFG_STATIC_LIB_NAME_x86_64-unknown-dragonfly=lib$(1).a
CFG_LIB_GLOB_x86_64-unknown-dragonfly=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_x86_64-unknown-dragonfly=$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_x86_64-unknown-dragonfly := -m64 -I/usr/include -I/usr/local/include $(CFLAGS)
CFG_GCCISH_CFLAGS_x86_64-unknown-dragonfly :=  -g -fPIC -m64 -I/usr/include -I/usr/local/include $(CFLAGS)
CFG_GCCISH_LINK_FLAGS_x86_64-unknown-dragonfly := -shared -fPIC -g -pthread  -lrt -m64
CFG_GCCISH_DEF_FLAG_x86_64-unknown-dragonfly := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_x86_64-unknown-dragonfly :=
CFG_INSTALL_NAME_x86_64-unknown-dragonfly =
CFG_EXE_SUFFIX_x86_64-unknown-dragonfly :=
CFG_WINDOWSY_x86_64-unknown-dragonfly :=
CFG_UNIXY_x86_64-unknown-dragonfly := 1
CFG_LDPATH_x86_64-unknown-dragonfly :=
CFG_RUN_x86_64-unknown-dragonfly=$(2)
CFG_RUN_TARG_x86_64-unknown-dragonfly=$(call CFG_RUN_x86_64-unknown-dragonfly,,$(2))
CFG_GNU_TRIPLE_x86_64-unknown-dragonfly := x86_64-unknown-dragonfly
