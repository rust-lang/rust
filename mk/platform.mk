# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.


# Create variables HOST_<triple> containing the host part
# of each target triple.  For example, the triple i686-darwin-macos
# would create a variable HOST_i686-darwin-macos with the value 
# i386.
define DEF_HOST_VAR
  HOST_$(1) = $(subst i686,i386,$(word 1,$(subst -, ,$(1))))
endef
$(foreach t,$(CFG_TARGET_TRIPLES),$(eval $(call DEF_HOST_VAR,$(t))))
$(foreach t,$(CFG_TARGET_TRIPLES),$(info cfg: host for $(t) is $(HOST_$(t))))

# Ditto for OSTYPE
define DEF_OSTYPE_VAR
  OSTYPE_$(1) = $(subst $(firstword $(subst -, ,$(1)))-,,$(1))
endef
$(foreach t,$(CFG_TARGET_TRIPLES),$(eval $(call DEF_OSTYPE_VAR,$(t))))
$(foreach t,$(CFG_TARGET_TRIPLES),$(info cfg: os for $(t) is $(OSTYPE_$(t))))

# FIXME: no-omit-frame-pointer is just so that task_start_wrapper
# has a frame pointer and the stack walker can understand it. Turning off
# frame pointers everywhere is overkill
CFG_GCCISH_CFLAGS += -fno-omit-frame-pointer

# On Darwin, we need to run dsymutil so the debugging information ends
# up in the right place.  On other platforms, it automatically gets
# embedded into the executable, so use a no-op command.
CFG_DSYMUTIL := true

# x86_64-unknown-linux-gnu configuration
CFG_LIB_NAME_x86_64-unknown-linux-gnu=lib$(1).so
CFG_LIB_GLOB_x86_64-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_x86_64-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_GCCISH_CFLAGS_x86_64-unknown-linux-gnu := -Wall -Werror -g -fPIC -m64
CFG_GCCISH_CXXFLAGS_x86_64-unknown-linux-gnu := -fno-rtti
CFG_GCCISH_LINK_FLAGS_x86_64-unknown-linux-gnu := -shared -fPIC -ldl -lpthread -lrt -g -m64
CFG_GCCISH_DEF_FLAG_x86_64-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_x86_64-unknown-linux-gnu := -Wl,-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_x86_64-unknown-linux-gnu := -Wl,-no-whole-archive -Wl,-znoexecstack
CFG_DEF_SUFFIX_x86_64-unknown-linux-gnu := .linux.def
CFG_INSTALL_NAME_x86_64-unknown-linux-gnu =
CFG_GCCISH_CROSS_x86_64-unknown-linux-gnu =

# i686-unknown-linux-gnu configuration
CFG_LIB_NAME_i686-unknown-linux-gnu=lib$(1).so
CFG_LIB_GLOB_i686-unknown-linux-gnu=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_i686-unknown-linux-gnu=lib$(1)-*.dylib.dSYM
CFG_GCCISH_CFLAGS_i686-unknown-linux-gnu := -Wall -Werror -g -fPIC -m32
CFG_GCCISH_CXXFLAGS_i686-unknown-linux-gnu := -fno-rtti
CFG_GCCISH_LINK_FLAGS_i6868-unknown-linux-gnu := -shared -fPIC -ldl -lpthread -lrt -g -m32
CFG_GCCISH_DEF_FLAG_i6868-unknown-linux-gnu := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_i686-unknown-linux-gnu := -Wl,-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_i686-unknown-linux-gnu := -Wl,-no-whole-archive -Wl,-znoexecstack
CFG_DEF_SUFFIX_i686-unknown-linux-gnu := .linux.def
CFG_INSTALL_NAME_i686-unknown-linux-gnu =
CFG_GCCISH_CROSS_i686-unknown-linux-gnu =

# x86_64-apple-darwin configuration
CFG_LIB_NAME_x86_64-apple-darwin=lib$(1).dylib
CFG_LIB_GLOB_x86_64-apple-darwin=lib$(1)-*.dylib
CFG_LIB_DSYM_GLOB_x86_64-apple-darwin=lib$(1)-*.dylib.dSYM
CFG_GCCISH_CFLAGS_x86_64-apple-darwin := -Wall -Werror -g -fPIC -m64 -arch x86_64
CFG_GCCISH_CXXFLAGS_x86_64-apple-darwin := -fno-rtti
CFG_GCCISH_LINK_FLAGS_x86_64-apple-darwin := -dynamiclib -lpthread -framework CoreServices -Wl,-no_compact_unwind -m64
CFG_GCCISH_DEF_FLAG_x86_64-apple-darwin := -Wl,-exported_symbols_list,
CFG_GCCISH_PRE_LIB_FLAGS_x86_64-apple-darwin :=
CFG_GCCISH_POST_LIB_FLAGS_x86_64-apple-darwin :=
CFG_DEF_SUFFIX_x86_64-apple-darwin := .darwin.def
CFG_INSTALL_NAME_x86_64-apple-darwin = -Wl,-install_name,@rpath/$(1)

# i686-apple-darwin configuration
CFG_LIB_NAME_i686-apple-darwin=lib$(1).dylib
CFG_LIB_GLOB_i686-apple-darwin=lib$(1)-*.dylib
CFG_LIB_DSYM_GLOB_i686-apple-darwin=lib$(1)-*.dylib.dSYM
CFG_GCCISH_CFLAGS_i686-apple-darwin := -Wall -Werror -g -fPIC -m32 -arch i386
CFG_GCCISH_CXXFLAGS_i686-apple-darwin := -fno-rtti
CFG_GCCISH_LINK_FLAGS_i686-apple-darwin := -dynamiclib -lpthread -framework CoreServices -Wl,-no_compact_unwind -m32
CFG_GCCISH_DEF_FLAG_i686-apple-darwin := -Wl,-exported_symbols_list,
CFG_GCCISH_PRE_LIB_FLAGS_i686-apple-darwin :=
CFG_GCCISH_POST_LIB_FLAGS_i686-apple-darwin :=
CFG_DEF_SUFFIX_i686-apple-darwin := .darwin.def
CFG_INSTALL_NAME_i686-apple-darwin = -Wl,-install_name,@rpath/$(1)

# arm-unknown-android configuration
CFG_LIB_NAME_arm-unknown-android=lib$(1).so
CFG_LIB_GLOB_arm-unknown-android=lib$(1)-*.so
CFG_LIB_DSYM_GLOB_arm-unknown-android=lib$(1)-*.dylib.dSYM
CFG_GCCISH_CFLAGS_arm-unknown-android := -Wall -g -fPIC -D__arm__ -DANDROID -D__ANDROID__
CFG_GCCISH_CXXFLAGS_arm-unknown-android := -fno-rtti
CFG_GCCISH_LINK_FLAGS_arm-unknown-android := -shared -fPIC -ldl -g -lm -lsupc++ -lgnustl_shared
CFG_GCCISH_DEF_FLAG_arm-unknown-android := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_arm-unknown-android := -Wl,-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_arm-unknown-android := -Wl,-no-whole-archive -Wl,-znoexecstack
CFG_DEF_SUFFIX_arm-unknown-android := .android.def
CFG_INSTALL_NAME_arm-unknown-android =
CFG_GCCISH_CROSS_arm-unknown-android = $(CFG_CROSS_PREFIX_arm)

# i686-pc-mingw32 configuration
CFG_LIB_NAME_i686-pc-mingw32=$(1).dll
CFG_LIB_GLOB_i686-pc-mingw32=$(1)-*.dll
CFG_LIB_DSYM_GLOB_i686-pc-mingw32=$(1)-*.dylib.dSYM
CFG_GCCISH_CFLAGS_i686-pc-mingw32 := -Wall -Werror -g -march=i686
CFG_GCCISH_CXXFLAGS_i686-pc-mingw32 := -fno-rtti
CFG_GCCISH_LINK_FLAGS_i686-pc-mingw32 := -shared -fPIC -g
CFG_GCCISH_DEF_FLAG_i686-pc-mingw32 :=
CFG_GCCISH_PRE_LIB_FLAGS_i686-pc-mingw32 := 
CFG_GCCISH_POST_LIB_FLAGS_i686-pc-mingw32 := 
CFG_DEF_SUFFIX_i686-pc-mingw32 := .def
CFG_INSTALL_NAME_i686-pc-mingw32 =
CFG_GCCISH_CROSS_i686-pc-mingw32 =

# x86_64-unknown-freebsd configuration
CFG_LIB_NAME_x86_64-unknown-freebsd=lib$(1).so
CFG_LIB_GLOB_x86_64-unknown-freebsd=lib$(1)-*.so
CFG_GCCISH_CFLAGS_x86_64-unknown-freebsd += -fPIC -I/usr/local/include
CFG_GCCISH_LINK_FLAGS_x86_64-unknown-freebsd += -shared -fPIC -lpthread -lrt
CFG_GCCISH_DEF_FLAG_x86_64-unknown-freebsd := -Wl,--export-dynamic,--dynamic-list=
CFG_GCCISH_PRE_LIB_FLAGS_x86_64-unknown-freebsd := -Wl,-whole-archive
CFG_GCCISH_POST_LIB_FLAGS_x86_64-unknown-freebsd := -Wl,-no-whole-archive
CFG_DEF_SUFFIX_x86_64-unknown-freebsd := .bsd.def
CFG_INSTALL_NAME_x86_64-unknown-freebsd =
CFG_GCCISH_CROSS_x86_64-unknown-freebsd =

# Hack: not sure how to test if a file exists in make other than this
OS_SUPP = $(patsubst %,--suppressions=%,\
      $(wildcard $(CFG_SRC_DIR)src/etc/$(CFG_OSTYPE).supp*))

ifneq ($(findstring mingw,$(CFG_OSTYPE)),)
  CFG_WINDOWSY := 1
endif
ifneq ($(findstring linux,$(CFG_OSTYPE)),)
  CFG_UNIXY := 1
endif
ifneq ($(findstring darwin,$(CFG_OSTYPE)),)
  CFG_UNIXY := 1
endif
ifneq ($(findstring freebsd,$(CFG_OSTYPE)),)
  CFG_UNIXY := 1
endif

ifdef CFG_DISABLE_OPTIMIZE_CXX
  $(info cfg: disabling C++ optimization (CFG_DISABLE_OPTIMIZE_CXX))
  CFG_GCCISH_CFLAGS += -O0
else
  CFG_GCCISH_CFLAGS += -O2
endif

CFG_TESTLIB=$(CFG_BUILD_DIR)/$(2)/$(strip \
 $(if $(findstring stage0,$(1)), \
       stage0/$(CFG_LIBDIR), \
      $(if $(findstring stage1,$(1)), \
           stage1/$(CFG_LIBDIR), \
          $(if $(findstring stage2,$(1)), \
               stage2/$(CFG_LIBDIR), \
               $(if $(findstring stage3,$(1)), \
                    stage3/$(CFG_LIBDIR), \
               )))))/rustc/$(CFG_HOST_TRIPLE)/$(CFG_LIBDIR)

ifneq ($(findstring linux,$(CFG_OSTYPE)),)
  # -znoexecstack is here because librt is for some reason being created
  # with executable stack and Fedora (or SELinux) doesn't like that (#798)
  ifdef CFG_PERF
    ifneq ($(CFG_PERF_WITH_LOGFD),)
        CFG_PERF_TOOL := $(CFG_PERF) stat -r 3 --log-fd 2
    else
        CFG_PERF_TOOL := $(CFG_PERF) stat -r 3
    endif
  else
    ifdef CFG_VALGRIND
      CFG_PERF_TOOL :=\
        $(CFG_VALGRIND) --tool=cachegrind --cache-sim=yes --branch-sim=yes
    else
      CFG_PERF_TOOL := /usr/bin/time --verbose
    endif
  endif
  # Linux requires LLVM to be built like this to get backtraces into Rust code
  CFG_LLVM_BUILD_ENV="CXXFLAGS=-fno-omit-frame-pointer"
endif

ifdef CFG_UNIXY
  CFG_INFO := $(info cfg: unix-y environment)

  CFG_PATH_MUNGE := true
  CFG_EXE_SUFFIX :=
  CFG_LDPATH :=
  CFG_RUN=$(2)
  CFG_RUN_TARG=$(call CFG_RUN,,$(2))
  CFG_RUN_TEST=$(call CFG_RUN,,$(CFG_VALGRIND) $(1))
  CFG_LIBUV_LINK_FLAGS=

  ifdef CFG_ENABLE_MINGW_CROSS
    CFG_WINDOWSY := 1
    CFG_INFO := $(info cfg: mingw-cross)
    CFG_GCCISH_CROSS := i586-mingw32msvc-
    ifdef CFG_VALGRIND
      CFG_VALGRIND += wine
    endif

    CFG_GCCISH_CFLAGS := -march=i586
    CFG_GCCISH_PRE_LIB_FLAGS :=
    CFG_GCCISH_POST_LIB_FLAGS :=
    CFG_GCCISH_DEF_FLAG :=
    CFG_GCCISH_LINK_FLAGS := -shared

    ifeq ($(CFG_CPUTYPE), x86_64)
      CFG_GCCISH_CFLAGS += -m32
      CFG_GCCISH_LINK_FLAGS += -m32
    endif
  endif
  ifdef CFG_VALGRIND
    CFG_VALGRIND += --error-exitcode=100 \
                    --quiet \
                    --suppressions=$(CFG_SRC_DIR)src/etc/x86.supp \
                    $(OS_SUPP)
    ifdef CFG_ENABLE_HELGRIND
      CFG_VALGRIND += --tool=helgrind
    else
      CFG_VALGRIND += --tool=memcheck \
                      --leak-check=full
    endif
  endif
endif

ifdef CFG_WINDOWSY
  CFG_INFO := $(info cfg: windows-y environment)

  CFG_EXE_SUFFIX := .exe
ifdef MSYSTEM
  CFG_LDPATH :=$(CFG_LDPATH):$$PATH
  CFG_RUN=PATH="$(CFG_LDPATH):$(1)" $(2)
else
  CFG_LDPATH :=
  CFG_RUN=$(2)
endif
  CFG_RUN_TARG=$(call CFG_RUN,$(HLIB$(1)_H_$(CFG_HOST_TRIPLE)),$(2))
  CFG_RUN_TEST=$(call CFG_RUN,$(call CFG_TESTLIB,$(1),$(3)),$(1))
  CFG_LIBUV_LINK_FLAGS=-lWs2_32 -lpsapi -liphlpapi

  ifndef CFG_ENABLE_MINGW_CROSS
    CFG_PATH_MUNGE := $(strip perl -i.bak -p             \
                             -e 's@\\(\S)@/\1@go;'       \
                             -e 's@^/([a-zA-Z])/@\1:/@o;')
  endif
endif

CFG_INFO := $(info cfg: using $(CFG_C_COMPILER))
ifeq ($(CFG_C_COMPILER),clang)
  ifeq ($(origin CC),default)
    CC=clang
  endif
  ifeq ($(origin CXX),default)
    CXX=clang++
  endif
  ifeq ($(origin CPP),default)
    CPP=clang
  endif
  CFG_GCCISH_CFLAGS += 
  CFG_GCCISH_CXXFLAGS += 
  CFG_GCCISH_LINK_FLAGS += 
  # These flags will cause the compiler to produce a .d file
  # next to the .o file that lists header deps.
  CFG_DEPEND_FLAGS = -MMD -MP -MT $(1) -MF $(1:%.o=%.d)

else
ifeq ($(CFG_C_COMPILER),gcc)
  ifeq ($(origin CC),default)
    CC=gcc
  endif
  ifeq ($(origin CXX),default)
    CXX=g++
  endif
  ifeq ($(origin CPP),default)
    CPP=gcc
  endif
  CFG_GCCISH_CFLAGS += 
  CFG_GCCISH_CXXFLAGS += 
  CFG_GCCISH_LINK_FLAGS += 
  # These flags will cause the compiler to produce a .d file
  # next to the .o file that lists header deps.
  CFG_DEPEND_FLAGS = -MMD -MP -MT $(1) -MF $(1:%.o=%.d)

else
  CFG_ERR := $(error please try on a system with gcc or clang)
endif
endif

define CFG_MAKE_CC
  ifeq ($$(CFG_BUILD_TRIPLE),$(1))
  
  CFG_COMPILE_C_$(1) = $$(CC)  \
        $$(CFG_GCCISH_CFLAGS)      \
        $$(CFG_GCCISH_CFLAGS_$(1)) \
        $$(CFG_DEPEND_FLAGS)       \
        -c -o $$(1) $$(2)
  CFG_LINK_C_$(1) = $$(CC) \
        $$(CFG_GCCISH_LINK_FLAGS) -o $$(1)          \
        $$(CFG_GCCISH_LINK_FLAGS_$(1)))             \
        $$(CFG_GCCISH_DEF_FLAG_$(1))$$(3) $$(2)     \
        $$(call CFG_INSTALL_NAME_$(1),$$(4))
  CFG_COMPILE_CXX_$(1) = $$(CXX) \
        $$(CFG_GCCISH_CFLAGS)      \
        $$(CFG_GCCISH_CXXFLAGS)    \
        $$(CFG_GCCISH_CFLAGS_$(1)) \
        $$(CFG_GCCISH_CXXFLAGS_$(1))    \
        $$(CFG_DEPEND_FLAGS)       \
        -c -o $$(1) $$(2)
  CFG_LINK_CXX_$(1) = $$(CXX) \
        $$(CFG_GCCISH_LINK_FLAGS) -o $$(1)             \
        $$(CFG_GCCISH_LINK_FLAGS_$(1))                 \
        $$(CFG_GCCISH_DEF_FLAG_$(1))$$(3) $$(2)        \
        $$(call CFG_INSTALL_NAME_$(1),$$(4))

  else
  
  CFG_COMPILE_C_$(1) = $(CFG_GCCISH_CROSS_$(1))$$(CC)  \
        $$(CFG_GCCISH_CFLAGS)      \
        $$(CFG_GCCISH_CFLAGS_$(1)) \
        $$(CFG_DEPEND_FLAGS)       \
        -c -o $$(1) $$(2)
  CFG_LINK_C_$(1) = $(CFG_GCCISH_CROSS_$(1))$$(CC) \
        $$(CFG_GCCISH_LINK_FLAGS) -o $$(1)          \
        $$(CFG_GCCISH_LINK_FLAGS_$(1)))             \
        $$(CFG_GCCISH_DEF_FLAG_$(1))$$(3) $$(2)     \
        $$(call CFG_INSTALL_NAME_$(1),$$(4))
  CFG_COMPILE_CXX_$(1) = $(CFG_GCCISH_CROSS_$(1))$$(CXX) \
        $$(CFG_GCCISH_CFLAGS)      \
        $$(CFG_GCCISH_CXXFLAGS)    \
        $$(CFG_GCCISH_CFLAGS_$(1)) \
        $$(CFG_GCCISH_CXXFLAGS_$(1))    \
        $$(CFG_DEPEND_FLAGS)       \
        -c -o $$(1) $$(2)
  CFG_LINK_CXX_$(1) = $(CFG_GCCISH_CROSS_$(1))$$(CXX) \
        $$(CFG_GCCISH_LINK_FLAGS) -o $$(1)             \
        $$(CFG_GCCISH_LINK_FLAGS_$(1))                 \
        $$(CFG_GCCISH_DEF_FLAG_$(1))$$(3) $$(2)        \
        $$(call CFG_INSTALL_NAME_$(1),$$(4))

  endif

endef

$(foreach target,$(CFG_TARGET_TRIPLES), \
  $(eval $(call CFG_MAKE_CC,$(target))))

# We're using llvm-mc as our assembler because it supports
# .cfi pseudo-ops on mac
define CFG_MAKE_ASSEMBLER
  ifeq ($$(CFG_BUILD_TRIPLE),$(1))

  CFG_ASSEMBLE_$(1)=$$(CPP) -E $$(CFG_DEPEND_FLAGS) $$(2) | \
                    $$(LLVM_MC_$$(CFG_BUILD_TRIPLE)) \
                    -assemble \
                    -filetype=obj \
                    -triple=$(1) \
                    -o=$$(1)
  else

  CFG_ASSEMBLE_$(1)=$(CFG_GCCISH_CROSS_$(1))$$(CPP) $$(CFG_DEPEND_FLAGS) $$(2) -c -o $$(1) 

  endif

endef

$(foreach target,$(CFG_TARGET_TRIPLES),\
  $(eval $(call CFG_MAKE_ASSEMBLER,$(target))))
