# armv7-unknown-linux-gnueabihf configuration
CROSS_PREFIX_armv7-unknown-linux-gnueabihf=arm-linux-gnueabihf-
CC_armv7-unknown-linux-gnueabihf=gcc
CXX_armv7-unknown-linux-gnueabihf=g++
CPP_armv7-unknown-linux-gnueabihf=gcc -E
AR_armv7-unknown-linux-gnueabihf=ar
CFG_LIB_NAME_armv7-unknown-linux-gnueabihf=lib$(1).so
CFG_STATIC_LIB_NAME_armv7-unknown-linux-gnueabihf=lib$(1).a
CFG_LIB_GLOB_armv7-unknown-linux-gnueabihf=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_armv7-unknown-linux-gnueabihf=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_armv7-unknown-linux-gnueabihf := -D__arm__ $(CFLAGS) -march=armv7-a
CFG_GCCISH_CFLAGS_armv7-unknown-linux-gnueabihf := -Wall -g -fPIC -D__arm__ $(CFLAGS) -march=armv7-a
CFG_GCCISH_CXXFLAGS_armv7-unknown-linux-gnueabihf := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_armv7-unknown-linux-gnueabihf := -shared -fPIC -g
CFG_GCCISH_DEF_FLAG_armv7-unknown-linux-gnueabihf := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_armv7-unknown-linux-gnueabihf :=
CFG_INSTALL_NAME_ar,-unknown-linux-gnueabihf =
CFG_EXE_SUFFIX_armv7-unknown-linux-gnueabihf :=
CFG_WINDOWSY_armv7-unknown-linux-gnueabihf :=
CFG_UNIXY_armv7-unknown-linux-gnueabihf := 1
CFG_LDPATH_armv7-unknown-linux-gnueabihf :=
CFG_RUN_armv7-unknown-linux-gnueabihf=$(2)
CFG_RUN_TARG_armv7-unknown-linux-gnueabihf=$(call CFG_RUN_armv7-unknown-linux-gnueabihf,,$(2))
RUSTC_FLAGS_armv7-unknown-linux-gnueabihf :=
RUSTC_CROSS_FLAGS_armv7-unknown-linux-gnueabihf :=
CFG_GNU_TRIPLE_armv7-unknown-linux-gnueabihf := armv7-unknown-linux-gnueabihf
