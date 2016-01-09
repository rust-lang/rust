# arm-unknown-linux-gnueabi configuration
CROSS_PREFIX_arm-unknown-linux-gnueabi=arm-linux-gnueabi-
CC_arm-unknown-linux-gnueabi=gcc
CXX_arm-unknown-linux-gnueabi=g++
CPP_arm-unknown-linux-gnueabi=gcc -E
AR_arm-unknown-linux-gnueabi=ar
CFG_LIB_NAME_arm-unknown-linux-gnueabi=lib$(1).so
CFG_STATIC_LIB_NAME_arm-unknown-linux-gnueabi=lib$(1).a
CFG_LIB_GLOB_arm-unknown-linux-gnueabi=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_arm-unknown-linux-gnueabi=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_arm-unknown-linux-gnueabi := -D__arm__ -mfloat-abi=soft $(CFLAGS)
CFG_GCCISH_CFLAGS_arm-unknown-linux-gnueabi := -Wall -g -fPIC -D__arm__ -mfloat-abi=soft $(CFLAGS)
CFG_GCCISH_CXXFLAGS_arm-unknown-linux-gnueabi := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_arm-unknown-linux-gnueabi := -shared -fPIC -g
CFG_GCCISH_DEF_FLAG_arm-unknown-linux-gnueabi := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_arm-unknown-linux-gnueabi :=
CFG_INSTALL_NAME_arm-unknown-linux-gnueabi =
CFG_EXE_SUFFIX_arm-unknown-linux-gnueabi :=
CFG_WINDOWSY_arm-unknown-linux-gnueabi :=
CFG_UNIXY_arm-unknown-linux-gnueabi := 1
CFG_LDPATH_arm-unknown-linux-gnueabi :=
CFG_RUN_arm-unknown-linux-gnueabi=$(2)
CFG_RUN_TARG_arm-unknown-linux-gnueabi=$(call CFG_RUN_arm-unknown-linux-gnueabi,,$(2))
RUSTC_FLAGS_arm-unknown-linux-gnueabi :=
RUSTC_CROSS_FLAGS_arm-unknown-linux-gnueabi :=
CFG_GNU_TRIPLE_arm-unknown-linux-gnueabi := arm-unknown-linux-gnueabi
