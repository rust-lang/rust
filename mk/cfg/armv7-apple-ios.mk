# armv7-apple-ios configuration
CFG_SDK_NAME_armv7-apple-ios := iphoneos
CFG_SDK_ARCHS_armv7-apple-ios := armv7
ifneq ($(findstring darwin,$(CFG_OSTYPE)),)
CFG_IOS_SDK_armv7-apple-ios := $(shell xcrun --show-sdk-path -sdk iphoneos 2>/dev/null)
CFG_IOS_SDK_FLAGS_armv7-apple-ios := -target armv7-apple-ios -isysroot $(CFG_IOS_SDK_armv7-apple-ios) -mios-version-min=7.0
CC_armv7-apple-ios = $(shell xcrun -find -sdk iphoneos clang)
CXX_armv7-apple-ios = $(shell xcrun -find -sdk iphoneos clang++)
CPP_armv7-apple-ios = $(shell xcrun -find -sdk iphoneos clang++)
AR_armv7-apple-ios = $(shell xcrun -find -sdk iphoneos ar)
endif
CFG_LIB_NAME_armv7-apple-ios = lib$(1).a
CFG_LIB_GLOB_armv7-apple-ios = lib$(1)-*.a
CFG_INSTALL_ONLY_RLIB_armv7-apple-ios = 1
CFG_STATIC_LIB_NAME_armv7-apple-ios=lib$(1).a
CFG_LIB_DSYM_GLOB_armv7-apple-ios = lib$(1)-*.a.dSYM
CFG_JEMALLOC_CFLAGS_armv7-apple-ios := -arch armv7 -mfpu=vfp3 $(CFG_IOS_SDK_FLAGS_armv7-apple-ios)
CFG_GCCISH_CFLAGS_armv7-apple-ios :=  -g -fPIC $(CFG_IOS_SDK_FLAGS_armv7-apple-ios) -mfpu=vfp3 -arch armv7
CFG_GCCISH_CXXFLAGS_armv7-apple-ios := -fno-rtti $(CFG_IOS_SDK_FLAGS_armv7-apple-ios) -I$(CFG_IOS_SDK_armv7-apple-ios)/usr/include/c++/4.2.1
CFG_GCCISH_LINK_FLAGS_armv7-apple-ios := -lpthread -syslibroot $(CFG_IOS_SDK_armv7-apple-ios) -Wl,-no_compact_unwind
CFG_GCCISH_DEF_FLAG_armv7-apple-ios := -Wl,-exported_symbols_list,
CFG_LLC_FLAGS_armv7-apple-ios := -mattr=+vfp3,+v7,+neon -march=arm
CFG_INSTALL_NAME_armv7-apple-ios = -Wl,-install_name,@rpath/$(1)
CFG_EXE_SUFFIX_armv7-apple-ios :=
CFG_WINDOWSY_armv7-apple-ios :=
CFG_UNIXY_armv7-apple-ios := 1
CFG_LDPATH_armv7-apple-ios :=
CFG_RUN_armv7-apple-ios = $(2)
CFG_RUN_TARG_armv7-apple-ios = $(call CFG_RUN_armv7-apple-ios,,$(2))
CFG_GNU_TRIPLE_armv7-apple-ios := armv7-apple-ios
