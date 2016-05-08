# armv7-linux-androideabi configuration
CC_armv7-linux-androideabi=$(CFG_ARMV7_LINUX_ANDROIDEABI_NDK)/bin/arm-linux-androideabi-gcc
CXX_armv7-linux-androideabi=$(CFG_ARMV7_LINUX_ANDROIDEABI_NDK)/bin/arm-linux-androideabi-g++
CPP_armv7-linux-androideabi=$(CFG_ARMV7_LINUX_ANDROIDEABI_NDK)/bin/arm-linux-androideabi-gcc -E
AR_armv7-linux-androideabi=$(CFG_ARMV7_LINUX_ANDROIDEABI_NDK)/bin/arm-linux-androideabi-ar
CFG_LIB_NAME_armv7-linux-androideabi=lib$(1).so
CFG_STATIC_LIB_NAME_armv7-linux-androideabi=lib$(1).a
CFG_LIB_GLOB_armv7-linux-androideabi=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_armv7-linux-androideabi=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_armv7-linux-androideabi := -D__arm__ -DANDROID -D__ANDROID__ $(CFLAGS)
CFG_GCCISH_CFLAGS_armv7-linux-androideabi := -Wall -g -fPIC -D__arm__ -mfloat-abi=softfp -march=armv7-a -mfpu=vfpv3-d16 -DANDROID -D__ANDROID__ $(CFLAGS)
CFG_GCCISH_CXXFLAGS_armv7-linux-androideabi := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_armv7-linux-androideabi := -shared -fPIC -ldl -g -lm -lsupc++
CFG_GCCISH_DEF_FLAG_armv7-linux-androideabi := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_armv7-linux-androideabi :=
CFG_INSTALL_NAME_armv7-linux-androideabi =
CFG_EXE_SUFFIX_armv7-linux-androideabi :=
CFG_WINDOWSY_armv7-linux-androideabi :=
CFG_UNIXY_armv7-linux-androideabi := 1
CFG_LDPATH_armv7-linux-androideabi :=
CFG_RUN_armv7-linux-androideabi=
CFG_RUN_TARG_armv7-linux-androideabi=
RUSTC_FLAGS_armv7-linux-androideabi :=
RUSTC_CROSS_FLAGS_armv7-linux-androideabi :=
CFG_GNU_TRIPLE_armv7-linux-androideabi := arm-linux-androideabi
