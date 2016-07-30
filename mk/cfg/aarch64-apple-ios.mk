# aarch64-apple-ios configuration
CFG_SDK_NAME_aarch64-apple-ios := iphoneos
CFG_SDK_ARCHS_aarch64-apple-ios := arm64
ifneq ($(findstring darwin,$(CFG_OSTYPE)),)
CFG_IOS_SDK_aarch64-apple-ios := $(shell xcrun --show-sdk-path -sdk iphoneos 2>/dev/null)
CFG_IOS_SDK_FLAGS_aarch64-apple-ios := -target aarch64-apple-darwin -isysroot $(CFG_IOS_SDK_aarch64-apple-ios) -mios-version-min=7.0 -arch arm64
CC_aarch64-apple-ios = $(shell xcrun -find -sdk iphoneos clang)
LINK_aarch64-apple-ios = $(shell xcrun -find -sdk iphoneos clang)
CXX_aarch64-apple-ios = $(shell xcrun -find -sdk iphoneos clang++)
CPP_aarch64-apple-ios = $(shell xcrun -find -sdk iphoneos clang++)
AR_aarch64-apple-ios = $(shell xcrun -find -sdk iphoneos ar)
endif
CFG_LIB_NAME_aarch64-apple-ios = lib$(1).a
CFG_LIB_GLOB_aarch64-apple-ios = lib$(1)-*.a
CFG_INSTALL_ONLY_RLIB_aarch64-apple-ios = 1
CFG_STATIC_LIB_NAME_aarch64-apple-ios=lib$(1).a
CFG_LIB_DSYM_GLOB_aarch64-apple-ios = lib$(1)-*.a.dSYM
CFG_CFLAGS_aarch64-apple-ios := $(CFG_IOS_SDK_FLAGS_aarch64-apple-ios)
CFG_JEMALLOC_CFLAGS_aarch64-apple-ios := $(CFG_IOS_SDK_FLAGS_aarch64-apple-ios)
CFG_GCCISH_CFLAGS_aarch64-apple-ios :=  -fPIC $(CFG_IOS_SDK_FLAGS_aarch64-apple-ios)
CFG_GCCISH_CXXFLAGS_aarch64-apple-ios := -fno-rtti $(CFG_IOS_SDK_FLAGS_aarch64-apple-ios) -I$(CFG_IOS_SDK_aarch64-apple-ios)/usr/include/c++/4.2.1
CFG_GCCISH_LINK_FLAGS_aarch64-apple-ios := -lpthread -syslibroot $(CFG_IOS_SDK_aarch64-apple-ios) -Wl,-no_compact_unwind
CFG_GCCISH_DEF_FLAG_aarch64-apple-ios := -Wl,-exported_symbols_list,
CFG_LLC_FLAGS_aarch64-apple-ios := -mattr=+neon,+cyclone,+fp-armv8
CFG_INSTALL_NAME_aarch64-apple-ios = -Wl,-install_name,@rpath/$(1)
CFG_LIBUV_LINK_FLAGS_aarch64-apple-ios =
CFG_EXE_SUFFIX_aarch64-apple-ios :=
CFG_WINDOWSY_aarch64-apple-ios :=
CFG_UNIXY_aarch64-apple-ios := 1
CFG_LDPATH_aarch64-apple-ios :=
CFG_RUN_aarch64-apple-ios = $(2)
CFG_RUN_TARG_aarch64-apple-ios = $(call CFG_RUN_aarch64-apple-ios,,$(2))
CFG_GNU_TRIPLE_aarch64-apple-ios := aarch64-apple-ios
