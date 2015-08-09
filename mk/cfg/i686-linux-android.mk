# i686-linux-android configuration
CC_i686-linux-android=$(CFG_I686_LINUX_ANDROID_NDK)/bin/i686-linux-android-gcc
CXX_i686-linux-android=$(CFG_I686_LINUX_ANDROID_NDK)/bin/i686-linux-android-g++
CPP_i686-linux-android=$(CFG_I686_LINUX_ANDROID_NDK)/bin/i686-linux-android-gcc -E
AR_i686-linux-android=$(CFG_I686_LINUX_ANDROID_NDK)/bin/i686-linux-android-ar
CFG_LIB_NAME_i686-linux-android=lib$(1).so
CFG_STATIC_LIB_NAME_i686-linux-android=lib$(1).a
CFG_LIB_GLOB_i686-linux-android=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_i686-linux-android=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_i686-linux-android := -D__i686__ -DANDROID -D__ANDROID__ $(CFLAGS)
CFG_GCCISH_CFLAGS_i686-linux-android := -Wall -g -fPIC -D__i686__ -DANDROID -D__ANDROID__ $(CFLAGS)
CFG_GCCISH_CXXFLAGS_i686-linux-android := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_i686-linux-android := -shared -fPIC -ldl -g -lm -lsupc++
CFG_GCCISH_DEF_FLAG_i686-linux-android := -Wl,--export-dynamic,--dynamic-list=
CFG_LLC_FLAGS_i686-linux-android :=
CFG_INSTALL_NAME_i686-linux-android =
CFG_EXE_SUFFIX_i686-linux-android :=
CFG_WINDOWSY_i686-linux-android :=
CFG_UNIXY_i686-linux-android := 1
CFG_LDPATH_i686-linux-android :=
CFG_RUN_i686-linux-android=
CFG_RUN_TARG_i686-linux-android=
RUSTC_FLAGS_i686-linux-android :=
RUSTC_CROSS_FLAGS_i686-linux-android :=
CFG_GNU_TRIPLE_i686-linux-android := i686-linux-android
