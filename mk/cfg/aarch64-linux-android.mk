# aarch64-linux-android configuration
# CROSS_PREFIX_aarch64-linux-android-
CC_aarch64-linux-android=$(CFG_ANDROID_CROSS_PATH)/bin/aarch64-linux-android-gcc
CXX_aarch64-linux-android=$(CFG_ANDROID_CROSS_PATH)/bin/aarch64-linux-android-g++
CPP_aarch64-linux-android=$(CFG_ANDROID_CROSS_PATH)/bin/aarch64-linux-android-gcc -E
AR_aarch64-linux-android=$(CFG_ANDROID_CROSS_PATH)/bin/aarch64-linux-android-ar
CFG_LIB_NAME_aarch64-linux-android=lib$(1).so
CFG_STATIC_LIB_NAME_aarch64-linux-android=lib$(1).a
CFG_LIB_GLOB_aarch64-linux-android=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_aarch64-linux-android=lib$(1)-*.dylib.dSYM
CFG_JEMALLOC_CFLAGS_aarch64-linux-android := -D__aarch64__ -DANDROID -D__ANDROID__ $(CFLAGS)
CFG_GCCISH_CFLAGS_aarch64-linux-android := -Wall -g -fPIC -D__aarch64__ -DANDROID -D__ANDROID__ $(CFLAGS)
CFG_GCCISH_CXXFLAGS_aarch64-linux-android := -fno-rtti $(CXXFLAGS)
CFG_GCCISH_LINK_FLAGS_aarch64-linux-android := -shared -fPIC -ldl -g -lm -lsupc++
CFG_GCCISH_DEF_FLAG_aarch64-linux-android := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_aarch64-linux-android := -Wl,-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_aarch64-linux-android := -Wl,-no-whole-archive
CFG_DEF_SUFFIX_aarch64-linux-android := .android.def
CFG_LLC_FLAGS_aarch64-linux-android :=
CFG_INSTALL_NAME_aarch64-linux-android =
CFG_EXE_SUFFIX_aarch64-linux-android :=
CFG_WINDOWSY_aarch64-linux-android :=
CFG_UNIXY_aarch64-linux-android := 1
CFG_PATH_MUNGE_aarch64-linux-android := true
CFG_LDPATH_aarch64-linux-android :=
CFG_RUN_aarch64-linux-android=
CFG_RUN_TARG_aarch64-linux-android=
RUSTC_FLAGS_aarch64-linux-android :=
RUSTC_CROSS_FLAGS_aarch64-linux-android :=
CFG_GNU_TRIPLE_aarch64-linux-android := aarch64-linux-android
