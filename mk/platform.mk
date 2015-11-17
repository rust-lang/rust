# Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
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
  HOST_$(1) = $(patsubst i%86,i386,$(word 1,$(subst -, ,$(1))))
endef
$(foreach t,$(CFG_TARGET),$(eval $(call DEF_HOST_VAR,$(t))))
$(foreach t,$(CFG_TARGET),$(info cfg: host for $(t) is $(HOST_$(t))))

# Ditto for OSTYPE
define DEF_OSTYPE_VAR
  OSTYPE_$(1) = $(subst $(firstword $(subst -, ,$(1)))-,,$(1))
endef
$(foreach t,$(CFG_TARGET),$(eval $(call DEF_OSTYPE_VAR,$(t))))
$(foreach t,$(CFG_TARGET),$(info cfg: os for $(t) is $(OSTYPE_$(t))))

# On Darwin, we need to run dsymutil so the debugging information ends
# up in the right place.  On other platforms, it automatically gets
# embedded into the executable, so use a no-op command.
CFG_DSYMUTIL := true

# Hack: not sure how to test if a file exists in make other than this
OS_SUPP = $(patsubst %,--suppressions=%, \
      $(wildcard $(CFG_SRC_DIR)src/etc/$(CFG_OSTYPE).supp*))

ifdef CFG_DISABLE_OPTIMIZE_CXX
  $(info cfg: disabling C++ optimization (CFG_DISABLE_OPTIMIZE_CXX))
  CFG_GCCISH_CFLAGS += -O0
else
  CFG_GCCISH_CFLAGS += -O2
endif

# The soname thing is for supporting a statically linked jemalloc.
# see https://blog.mozilla.org/jseward/2012/06/05/valgrind-now-supports-jemalloc-builds-directly/
ifdef CFG_VALGRIND
  CFG_VALGRIND += --error-exitcode=100 \
                  --fair-sched=try \
                  --quiet \
                  --soname-synonyms=somalloc=NONE \
                  --suppressions=$(CFG_SRC_DIR)src/etc/x86.supp \
                  $(OS_SUPP)
  ifdef CFG_ENABLE_HELGRIND
    CFG_VALGRIND += --tool=helgrind
  else
    CFG_VALGRIND += --tool=memcheck \
                    --leak-check=full
  endif
endif

# If we actually want to run Valgrind on a given platform, set this variable
define DEF_GOOD_VALGRIND
  ifeq ($(OSTYPE_$(1)),unknown-linux-gnu)
    GOOD_VALGRIND_$(1) = 1
  endif
  ifneq (,$(filter $(OSTYPE_$(1)),darwin freebsd))
    ifeq (HOST_$(1),x86_64)
      GOOD_VALGRIND_$(1) = 1
    endif
  endif
endef
$(foreach t,$(CFG_TARGET),$(eval $(call DEF_GOOD_VALGRIND,$(t))))
$(foreach t,$(CFG_TARGET),$(info cfg: good valgrind for $(t) is $(GOOD_VALGRIND_$(t))))

ifneq ($(findstring linux,$(CFG_OSTYPE)),)
  ifdef CFG_PERF
    ifneq ($(CFG_PERF_WITH_LOGFD),)
        CFG_PERF_TOOL := $(CFG_PERF) stat -r 3 --log-fd 2
    else
        CFG_PERF_TOOL := $(CFG_PERF) stat -r 3
    endif
  else
    ifdef CFG_VALGRIND
      CFG_PERF_TOOL := \
        $(CFG_VALGRIND) --tool=cachegrind --cache-sim=yes --branch-sim=yes
    else
      CFG_PERF_TOOL := /usr/bin/time --verbose
    endif
  endif
endif

AR := ar

define SET_FROM_CFG
  ifdef CFG_$(1)
    ifeq ($(origin $(1)),undefined)
      $$(info cfg: using $(1)=$(CFG_$(1)) (CFG_$(1)))
      $(1)=$(CFG_$(1))
    endif
    ifeq ($(origin $(1)),default)
      $$(info cfg: using $(1)=$(CFG_$(1)) (CFG_$(1)))
      $(1)=$(CFG_$(1))
    endif
  endif
endef

$(foreach cvar,CC CXX CPP CFLAGS CXXFLAGS CPPFLAGS, \
  $(eval $(call SET_FROM_CFG,$(cvar))))

CFG_RLIB_GLOB=lib$(1)-*.rlib

include $(wildcard $(CFG_SRC_DIR)mk/cfg/*.mk)

define ADD_INSTALLED_OBJECTS
  INSTALLED_OBJECTS_$(1) += $$(CFG_INSTALLED_OBJECTS_$(1))
  REQUIRED_OBJECTS_$(1) += $$(CFG_THIRD_PARTY_OBJECTS_$(1))
  INSTALLED_OBJECTS_$(1) += $$(call CFG_STATIC_LIB_NAME_$(1),compiler-rt)
  REQUIRED_OBJECTS_$(1) += $$(call CFG_STATIC_LIB_NAME_$(1),compiler-rt)
endef

$(foreach target,$(CFG_TARGET), \
  $(eval $(call ADD_INSTALLED_OBJECTS,$(target))))

define DEFINE_LINKER
  ifndef LINK_$(1)
    LINK_$(1) := $$(CC_$(1))
  endif
endef

$(foreach target,$(CFG_TARGET), \
  $(eval $(call DEFINE_LINKER,$(target))))

# The -Qunused-arguments sidesteps spurious warnings from clang
define FILTER_FLAGS
  ifeq ($$(CFG_USING_CLANG),1)
    ifneq ($(findstring clang,$$(shell $(CC_$(1)) -v)),)
      CFG_GCCISH_CFLAGS_$(1) += -Qunused-arguments
      CFG_GCCISH_CXXFLAGS_$(1) += -Qunused-arguments
    endif
  endif
endef

$(foreach target,$(CFG_TARGET), \
  $(eval $(call FILTER_FLAGS,$(target))))

# Configure various macros to pass gcc or cl.exe style arguments
define CC_MACROS
  CFG_CC_INCLUDE_$(1)=-I $$(1)
  ifeq ($$(findstring msvc,$(1)),msvc)
    CFG_CC_OUTPUT_$(1)=-Fo:$$(1)
    CFG_CREATE_ARCHIVE_$(1)=$$(AR_$(1)) -OUT:$$(1)
  else
    CFG_CC_OUTPUT_$(1)=-o $$(1)
    CFG_CREATE_ARCHIVE_$(1)=$$(AR_$(1)) crus $$(1)
  endif
endef

$(foreach target,$(CFG_TARGET), \
  $(eval $(call CC_MACROS,$(target))))


ifeq ($(CFG_CCACHE_CPP2),1)
  CCACHE_CPP2=1
  export CCACHE_CPP
endif

ifdef CFG_CCACHE_BASEDIR
  CCACHE_BASEDIR=$(CFG_CCACHE_BASEDIR)
  export CCACHE_BASEDIR
endif

FIND_COMPILER = $(word 1,$(1:ccache=))

define CFG_MAKE_TOOLCHAIN
  # Prepend the tools with their prefix if cross compiling
  ifneq ($(CFG_BUILD),$(1))
    ifneq ($$(findstring msvc,$(1)),msvc)
       CC_$(1)=$(CROSS_PREFIX_$(1))$(CC_$(1))
       CXX_$(1)=$(CROSS_PREFIX_$(1))$(CXX_$(1))
       CPP_$(1)=$(CROSS_PREFIX_$(1))$(CPP_$(1))
       AR_$(1)=$(CROSS_PREFIX_$(1))$(AR_$(1))
       LINK_$(1)=$(CROSS_PREFIX_$(1))$(LINK_$(1))
       RUSTC_CROSS_FLAGS_$(1)=-C linker=$$(call FIND_COMPILER,$$(LINK_$(1))) \
           -C ar=$$(call FIND_COMPILER,$$(AR_$(1))) $(RUSTC_CROSS_FLAGS_$(1))

       RUSTC_FLAGS_$(1)=$$(RUSTC_CROSS_FLAGS_$(1)) $(RUSTC_FLAGS_$(1))
    endif
  endif

  CFG_COMPILE_C_$(1) = $$(CC_$(1)) \
        $$(CFG_GCCISH_CFLAGS) \
        $$(CFG_GCCISH_CFLAGS_$(1)) \
        -c $$(call CFG_CC_OUTPUT_$(1),$$(1)) $$(2)
  CFG_LINK_C_$(1) = $$(CC_$(1)) \
        $$(CFG_GCCISH_LINK_FLAGS) -o $$(1) \
        $$(CFG_GCCISH_LINK_FLAGS_$(1)) \
        $$(CFG_GCCISH_DEF_FLAG_$(1))$$(3) $$(2) \
        $$(call CFG_INSTALL_NAME_$(1),$$(4))
  CFG_COMPILE_CXX_$(1) = $$(CXX_$(1)) \
        $$(CFG_GCCISH_CFLAGS) \
        $$(CFG_GCCISH_CXXFLAGS) \
        $$(CFG_GCCISH_CFLAGS_$(1)) \
        $$(CFG_GCCISH_CXXFLAGS_$(1)) \
        -c $$(call CFG_CC_OUTPUT_$(1),$$(1)) $$(2)
  CFG_LINK_CXX_$(1) = $$(CXX_$(1)) \
        $$(CFG_GCCISH_LINK_FLAGS) -o $$(1) \
        $$(CFG_GCCISH_LINK_FLAGS_$(1)) \
        $$(CFG_GCCISH_DEF_FLAG_$(1))$$(3) $$(2) \
        $$(call CFG_INSTALL_NAME_$(1),$$(4))

  ifeq ($$(findstring $(HOST_$(1)),arm aarch64 mips mipsel powerpc),)

  # On OpenBSD, we need to pass the path of libstdc++.so to the linker
  # (use path of libstdc++.a which is a known name for the same path)
  ifeq ($(OSTYPE_$(1)),unknown-openbsd)
    STDCPP_LIBDIR_RUSTFLAGS_$(1)= \
        -L "$$(dir $$(shell $$(CC_$(1)) $$(CFG_GCCISH_CFLAGS_$(1)) \
        -print-file-name=lib$(CFG_STDCPP_NAME).a))"
  else
    STDCPP_LIBDIR_RUSTFLAGS_$(1)=
  endif

  # On Bitrig, we need the relocation model to be PIC for everything
  ifeq (,$(filter $(OSTYPE_$(1)),bitrig))
    LLVM_MC_RELOCATION_MODEL="pic"
  else
    LLVM_MC_RELOCATION_MODEL="default"
  endif

  # We're using llvm-mc as our assembler because it supports
  # .cfi pseudo-ops on mac
  CFG_ASSEMBLE_$(1)=$$(CPP_$(1)) -E $$(2) | \
                    $$(LLVM_MC_$$(CFG_BUILD)) \
                    -assemble \
                    -relocation-model=$$(LLVM_MC_RELOCATION_MODEL) \
                    -filetype=obj \
                    -triple=$(1) \
                    -o=$$(1)
  else

  # For the ARM, AARCH64, MIPS and POWER crosses, use the toolchain assembler
  # FIXME: We should be able to use the LLVM assembler
  CFG_ASSEMBLE_$(1)=$$(CC_$(1)) $$(CFG_GCCISH_CFLAGS_$(1)) \
                   $$(2) -c -o $$(1)

  endif

endef

$(foreach target,$(CFG_TARGET), \
  $(eval $(call CFG_MAKE_TOOLCHAIN,$(target))))
