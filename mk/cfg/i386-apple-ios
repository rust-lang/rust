# i386-apple-ios configuration
CFG_SDK_NAME_i386-apple-ios = iphonesimulator
CFG_SDK_ARCHS_i386-apple-ios = i386
ifneq ($(findstring darwin,$(CFG_OSTYPE)),)
CFG_IOSSIM_SDK = $(shell xcrun --show-sdk-path -sdk iphonesimulator 2>/dev/null)
CFG_IOSSIM_FLAGS = -target i386-apple-ios -isysroot $(CFG_IOSSIM_SDK) -mios-simulator-version-min=7.0
CC_i386-apple-ios = $(shell xcrun -find -sdk iphonesimulator clang)
CXX_i386-apple-ios = $(shell xcrun -find -sdk iphonesimulator clang++)
CPP_i386-apple-ios = $(shell xcrun -find -sdk iphonesimulator clang++)
AR_i386-apple-ios = $(shell xcrun -find -sdk iphonesimulator ar)
endif
CFG_LIB_NAME_i386-apple-ios = lib$(1).a
CFG_LIB_GLOB_i386-apple-ios = lib$(1)-*.dylib
CFG_STATIC_LIB_NAME_i386-apple-ios=lib$(1).a
CFG_LIB_DSYM_GLOB_i386-apple-ios = lib$(1)-*.dylib.dSYM
CFG_GCCISH_CFLAGS_i386-apple-ios = -Wall -Werror -g -fPIC -m32 $(CFG_IOSSIM_FLAGS)
CFG_GCCISH_CXXFLAGS_i386-apple-ios = -fno-rtti $(CFG_IOSSIM_FLAGS) -I$(CFG_IOSSIM_SDK)/usr/include/c++/4.2.1
CFG_GCCISH_LINK_FLAGS_i386-apple-ios = -lpthread -Wl,-no_compact_unwind -m32 -Wl,-syslibroot $(CFG_IOSSIM_SDK)
CFG_GCCISH_DEF_FLAG_i386-apple-ios = -Wl,-exported_symbols_list,
CFG_GCCISH_PRE_LIB_FLAGS_i386-apple-ios =
CFG_GCCISH_POST_LIB_FLAGS_i386-apple-ios =
CFG_DEF_SUFFIX_i386-apple-ios = .darwin.def
CFG_LLC_FLAGS_i386-apple-ios =
CFG_INSTALL_NAME_i386-apple-ios = -Wl,-install_name,@rpath/$(1)
CFG_EXE_SUFFIX_i386-apple-ios =
CFG_WINDOWSY_i386-apple-ios =
CFG_UNIXY_i386-apple-ios = 1
CFG_PATH_MUNGE_i386-apple-ios = true
CFG_LDPATH_i386-apple-ios =
CFG_RUN_i386-apple-ios = $(2)
CFG_RUN_TARG_i386-apple-ios = $(call CFG_RUN_i386-apple-ios,,$(2))
CFG_JEMALLOC_CFLAGS_i386-apple-ios = $(CFG_IOSSIM_FLAGS) -target i386-apple-ios -Wl,-syslibroot $(CFG_IOSSIM_SDK) -Wl,-no_compact_unwind
CFG_GNU_TRIPLE_i386-apple-ios := i386-apple-ios
