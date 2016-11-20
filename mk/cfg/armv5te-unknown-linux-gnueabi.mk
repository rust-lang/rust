# armv5-unknown-linux-gnueabi configuration
CROSS_PREFIX_armv5te-unknown-linux-gnueabi=arm-linux-gnueabi-
CC_armv5te-unknown-linux-gnueabi=gcc
CXX_armv5te-unknown-linux-gnueabi=g++
CPP_armv5te-unknown-linux-gnueabi=gcc -E
AR_armv5te-unknown-linux-gnueabi=ar
CFG_LIB_NAME_armv5te-unknown-linux-gnueabi=lib$(1).so
CFG_STATIC_LIB_NAME_armv5te-unknown-linux-gnueabi=lib$(1).a
CFG_LIB_GLOB_armv5te-unknown-linux-gnueabi=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_armv5te-unknown-linux-gnueabi=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_armv5te-unknown-linux-gnueabi := -D__arm__ -mfloat-abi=soft  $(CFLAGS) -march=armv5te -marm
CFG_GCCISH_CFLAGS_armv5te-unknown-linux-gnueabi := -Wall -g -fPIC -D__arm__ -mfloat-abi=soft $(CFLAGS) -march=armv5te -marm
CFG_GCCISH_CXXFLAGS_armv5te-unknown-linux-gnueabi := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_armv5te-unknown-linux-gnueabi := -shared -fPIC -g
CFG_GCCISH_DEF_FLAG_armv5te-unknown-linux-gnueabi := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_armv5te-unknown-linux-gnueabi :=
CFG_INSTALL_NAME_ar,-unknown-linux-gnueabi =
CFG_EXE_SUFFIX_armv5te-unknown-linux-gnueabi :=
CFG_WINDOWSY_armv5te-unknown-linux-gnueabi :=
CFG_UNIXY_armv5te-unknown-linux-gnueabi := 1
CFG_LDPATH_armv5te-unknown-linux-gnueabi :=
CFG_RUN_armv5te-unknown-linux-gnueabi=$(2)
CFG_RUN_TARG_armv5te-unknown-linux-gnueabi=$(call CFG_RUN_armv5te-unknown-linux-gnueabi,,$(2))
RUSTC_FLAGS_armv5te-unknown-linux-gnueabi :=
RUSTC_CROSS_FLAGS_armv5te-unknown-linux-gnueabi :=
CFG_GNU_TRIPLE_armv5te-unknown-linux-gnueabi := armv5te-unknown-linux-gnueabi
