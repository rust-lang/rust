# arm-unknown-linux-gnueabihf configuration
CROSS_PREFIX_arm-unknown-linux-gnueabihf=arm-linux-gnueabihf-
CC_arm-unknown-linux-gnueabihf=gcc
CXX_arm-unknown-linux-gnueabihf=g++
CPP_arm-unknown-linux-gnueabihf=gcc -E
AR_arm-unknown-linux-gnueabihf=ar
CFG_LIB_NAME_arm-unknown-linux-gnueabihf=lib$(1).so
CFG_STATIC_LIB_NAME_arm-unknown-linux-gnueabihf=lib$(1).a
CFG_LIB_GLOB_arm-unknown-linux-gnueabihf=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_arm-unknown-linux-gnueabihf=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_arm-unknown-linux-gnueabihf := -D__arm__ $(CFLAGS)
CFG_GCCISH_CFLAGS_arm-unknown-linux-gnueabihf := -Wall -g -fPIC -D__arm__ $(CFLAGS)
CFG_GCCISH_CXXFLAGS_arm-unknown-linux-gnueabihf := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_arm-unknown-linux-gnueabihf := -shared -fPIC -g
CFG_GCCISH_DEF_FLAG_arm-unknown-linux-gnueabihf := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_arm-unknown-linux-gnueabihf := -Wl,-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_arm-unknown-linux-gnueabihf := -Wl,-no-whole-archive
CFG_DEF_SUFFIX_arm-unknown-linux-gnueabihf := .linux.def
CFG_LLC_FLAGS_arm-unknown-linux-gnueabihf :=
CFG_INSTALL_NAME_ar,-unknown-linux-gnueabihf =
CFG_EXE_SUFFIX_arm-unknown-linux-gnueabihf :=
CFG_WINDOWSY_arm-unknown-linux-gnueabihf :=
CFG_UNIXY_arm-unknown-linux-gnueabihf := 1
CFG_PATH_MUNGE_arm-unknown-linux-gnueabihf := true
CFG_LDPATH_arm-unknown-linux-gnueabihf :=
CFG_RUN_arm-unknown-linux-gnueabihf=$(2)
CFG_RUN_TARG_arm-unknown-linux-gnueabihf=$(call CFG_RUN_arm-unknown-linux-gnueabihf,,$(2))
RUSTC_FLAGS_arm-unknown-linux-gnueabihf := -C target-feature=+v6,+vfp2
RUSTC_CROSS_FLAGS_arm-unknown-linux-gnueabihf :=
CFG_GNU_TRIPLE_arm-unknown-linux-gnueabihf := arm-unknown-linux-gnueabihf
