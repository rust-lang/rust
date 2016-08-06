# arm-unknown-linux-musleabi configuration
CROSS_PREFIX_arm-unknown-linux-musleabi=arm-linux-musleabi-
CC_arm-unknown-linux-musleabi=gcc
CXX_arm-unknown-linux-musleabi=g++
CPP_arm-unknown-linux-musleabi=gcc -E
AR_arm-unknown-linux-musleabi=ar
CFG_LIB_NAME_arm-unknown-linux-musleabi=lib$(1).so
CFG_STATIC_LIB_NAME_arm-unknown-linux-musleabi=lib$(1).a
CFG_LIB_GLOB_arm-unknown-linux-musleabi=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_arm-unknown-linux-musleabi=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_arm-unknown-linux-musleabi := -D__arm__ -mfloat-abi=soft $(CFLAGS) -march=armv6 -marm
CFG_GCCISH_CFLAGS_arm-unknown-linux-musleabi := -Wall -g -fPIC -D__arm__ -mfloat-abi=soft $(CFLAGS) -march=armv6 -marm
CFG_GCCISH_CXXFLAGS_arm-unknown-linux-musleabi := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_arm-unknown-linux-musleabi := -shared -fPIC -g
CFG_GCCISH_DEF_FLAG_arm-unknown-linux-musleabi := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_arm-unknown-linux-musleabi :=
CFG_INSTALL_NAME_arm-unknown-linux-musleabi =
CFG_EXE_SUFFIX_arm-unknown-linux-musleabi :=
CFG_WINDOWSY_arm-unknown-linux-musleabi :=
CFG_UNIXY_arm-unknown-linux-musleabi := 1
CFG_LDPATH_arm-unknown-linux-musleabi :=
CFG_RUN_arm-unknown-linux-musleabi=$(2)
CFG_RUN_TARG_arm-unknown-linux-musleabi=$(call CFG_RUN_arm-unknown-linux-musleabi,,$(2))
RUSTC_FLAGS_arm-unknown-linux-musleabi :=
RUSTC_CROSS_FLAGS_arm-unknown-linux-musleabi :=
CFG_GNU_TRIPLE_arm-unknown-linux-musleabi := arm-unknown-linux-musleabi
