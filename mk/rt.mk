# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

################################################################################
# Native libraries built as part of the rust build process
#
# This portion of the rust build system is meant to keep track of native
# dependencies and how to build them. It is currently required that all native
# dependencies are built as static libraries, as slinging around dynamic
# libraries isn't exactly the most fun thing to do.
#
# This section should need minimal modification to add new libraries. The
# relevant variables are:
#
#   NATIVE_LIBS
#	This is a list of all native libraries which are built as part of the
#	build process. It will build all libraries into RT_OUTPUT_DIR with the
#	appropriate name of static library as dictated by the target platform
#
#   NATIVE_DEPS_<lib>
#	This is a list of files relative to the src/rt directory which are
#	needed to build the native library. Each file will be compiled to an
#	object file, and then all the object files will be assembled into an
#	archive (static library). The list contains files of any extension
#
# If adding a new library, you should update the NATIVE_LIBS list, and then list
# the required files below it. The list of required files is a list of files
# that's per-target so you're allowed to conditionally add files based on the
# target.
################################################################################
NATIVE_LIBS := rustrt sundown uv_support morestack miniz

# $(1) is the target triple
define NATIVE_LIBRARIES

NATIVE_DEPS_sundown_$(1) := sundown/src/autolink.c \
			sundown/src/buffer.c \
			sundown/src/stack.c \
			sundown/src/markdown.c \
			sundown/html/houdini_href_e.c \
			sundown/html/houdini_html_e.c \
			sundown/html/html_smartypants.c \
			sundown/html/html.c
NATIVE_DEPS_uv_support_$(1) := rust_uv.c
NATIVE_DEPS_miniz_$(1) = miniz.c
NATIVE_DEPS_rustrt_$(1) := rust_builtin.c \
			rust_android_dummy.c \
			rust_test_helpers.c \
			rust_try.ll \
			arch/$$(HOST_$(1))/_context.S \
			arch/$$(HOST_$(1))/record_sp.S
NATIVE_DEPS_morestack_$(1) := arch/$$(HOST_$(1))/morestack.S

################################################################################
# You shouldn't find it that necessary to edit anything below this line.
################################################################################

# While we're defining the native libraries for each target, we define some
# common rules used to build files for various targets.

RT_OUTPUT_DIR_$(1) := $(1)/rt

$$(RT_OUTPUT_DIR_$(1))/%.o: $(S)src/rt/%.ll $$(MKFILE_DEPS) \
	    $$(LLVM_CONFIG_$$(CFG_BUILD))
	@mkdir -p $$(@D)
	@$$(call E, compile: $$@)
	$$(Q)$$(LLC_$$(CFG_BUILD)) $$(CFG_LLC_FLAGS_$(1)) \
	    -filetype=obj -mtriple=$(1) -relocation-model=pic -o $$@ $$<

$$(RT_OUTPUT_DIR_$(1))/%.o: $(S)src/rt/%.c $$(MKFILE_DEPS)
	@mkdir -p $$(@D)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, \
		-I $$(S)src/rt/sundown/src -I $$(S)src/rt/sundown/html \
		-I $$(S)src/libuv/include -I $$(S)src/rt \
                 $$(RUNTIME_CFLAGS_$(1))) $$<

$$(RT_OUTPUT_DIR_$(1))/%.o: $(S)src/rt/%.S $$(MKFILE_DEPS) \
	    $$(LLVM_CONFIG_$$(CFG_BUILD))
	@mkdir -p $$(@D)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_ASSEMBLE_$(1),$$@,$$<)
endef

$(foreach target,$(CFG_TARGET),$(eval $(call NATIVE_LIBRARIES,$(target))))

# A macro for devining how to build third party libraries listed above (based
# on their dependencies).
#
# $(1) is the target
# $(2) is the lib name
define THIRD_PARTY_LIB

OBJS_$(2)_$(1) := $$(NATIVE_DEPS_$(2)_$(1):%=$$(RT_OUTPUT_DIR_$(1))/%)
OBJS_$(2)_$(1) := $$(OBJS_$(2)_$(1):.c=.o)
OBJS_$(2)_$(1) := $$(OBJS_$(2)_$(1):.cpp=.o)
OBJS_$(2)_$(1) := $$(OBJS_$(2)_$(1):.ll=.o)
OBJS_$(2)_$(1) := $$(OBJS_$(2)_$(1):.S=.o)
NATIVE_$(2)_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),$(2))
$$(RT_OUTPUT_DIR_$(1))/$$(NATIVE_$(2)_$(1)): $$(OBJS_$(2)_$(1))
	@$$(call E, link: $$@)
	$$(Q)$$(AR_$(1)) rcs $$@ $$^

endef

$(foreach target,$(CFG_TARGET),					    \
 $(eval $(call RUNTIME_RULES,$(target))))
$(foreach lib,$(NATIVE_LIBS),					    \
 $(foreach target,$(CFG_TARGET),				    \
  $(eval $(call THIRD_PARTY_LIB,$(target),$(lib)))))


################################################################################
# Building third-party targets with external build systems
#
# The only current member of this section is libuv, but long ago this used to
# also be occupied by jemalloc. This location is meant for dependencies which
# have external build systems. It is still assumed that the output of each of
# these steps is a static library in the correct location.
################################################################################

define DEF_LIBUV_ARCH_VAR
  LIBUV_ARCH_$(1) = $$(subst i386,ia32,$$(subst x86_64,x64,$$(HOST_$(1))))
endef
$(foreach t,$(CFG_TARGET),$(eval $(call DEF_LIBUV_ARCH_VAR,$(t))))

ifdef CFG_ENABLE_FAST_MAKE
LIBUV_DEPS := $(S)/.gitmodules
else
LIBUV_DEPS := $(wildcard \
              $(S)src/libuv/* \
              $(S)src/libuv/*/* \
              $(S)src/libuv/*/*/* \
              $(S)src/libuv/*/*/*/*)
endif

LIBUV_NO_LOAD = run-benchmarks.target.mk run-tests.target.mk \
		uv_dtrace_header.target.mk uv_dtrace_provider.target.mk

export PYTHONPATH := $(PYTHONPATH):$(S)src/gyp/pylib

define DEF_THIRD_PARTY_TARGETS

# $(1) is the target triple

ifeq ($$(CFG_WINDOWSY_$(1)), 1)
  LIBUV_OSTYPE_$(1) := win
else ifeq ($(OSTYPE_$(1)), apple-darwin)
  LIBUV_OSTYPE_$(1) := mac
else ifeq ($(OSTYPE_$(1)), unknown-freebsd)
  LIBUV_OSTYPE_$(1) := freebsd
else ifeq ($(OSTYPE_$(1)), linux-androideabi)
  LIBUV_OSTYPE_$(1) := android
  LIBUV_ARGS_$(1) := PLATFORM=android host=android OS=linux
else
  LIBUV_OSTYPE_$(1) := linux
endif

LIBUV_NAME_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),uv)
LIBUV_DIR_$(1) := $$(RT_OUTPUT_DIR_$(1))/libuv
LIBUV_LIB_$(1) := $$(RT_OUTPUT_DIR_$(1))/$$(LIBUV_NAME_$(1))

LIBUV_MAKEFILE_$(1) := $$(CFG_BUILD_DIR)$$(RT_OUTPUT_DIR_$(1))/libuv/Makefile

LIBUV_STAMP_$(1) = $$(LIBUV_DIR_$(1))/libuv-auto-clean-stamp

$$(LIBUV_STAMP_$(1)): $(S)src/rt/libuv-auto-clean-trigger
	$$(Q)rm -rf $$(LIBUV_DIR_$(1))
	$$(Q)mkdir -p $$(@D)
	touch $$@

# libuv triggers a few warnings on some platforms
LIBUV_CFLAGS_$(1) := $(subst -Werror,,$(CFG_GCCISH_CFLAGS_$(1)))

$$(LIBUV_MAKEFILE_$(1)): $$(LIBUV_DEPS) $$(MKFILE_DEPS) $$(LIBUV_STAMP_$(1))
	(cd $(S)src/libuv/ && \
	 $$(CFG_PYTHON) ./gyp_uv.py -f make -Dtarget_arch=$$(LIBUV_ARCH_$(1)) \
	   -D ninja \
	   -DOS=$$(LIBUV_OSTYPE_$(1)) \
	   -Goutput_dir=$$(@D) --generator-output $$(@D))
	touch $$@

# Windows has a completely different build system for libuv because of mingw. In
# theory when we support msvc then we should be using gyp's msvc output instead
# of mingw's makefile for windows
ifdef CFG_WINDOWSY_$(1)
$$(LIBUV_LIB_$(1)): $$(LIBUV_DEPS) $$(MKFILE_DEPS)
	$$(Q)$$(MAKE) -C $$(S)src/libuv -f Makefile.mingw \
		LDFLAGS="$$(CFG_GCCISH_LINK_FLAGS_$(1))" \
		CC="$$(CC_$(1)) $$(LIBUV_CFLAGS_$(1)) $$(SNAP_DEFINES)" \
		CXX="$$(CXX_$(1))" \
		AR="$$(AR_$(1))" \
		V=$$(VERBOSE)
	$$(Q)cp $$(S)src/libuv/libuv.a $$@
else
$$(LIBUV_LIB_$(1)): $$(LIBUV_DIR_$(1))/Release/libuv.a $$(MKFILE_DEPS)
	$$(Q)cp $$< $$@
$$(LIBUV_DIR_$(1))/Release/libuv.a: $$(LIBUV_DEPS) $$(LIBUV_MAKEFILE_$(1)) \
				    $$(MKFILE_DEPS)
	$$(Q)$$(MAKE) -C $$(LIBUV_DIR_$(1)) \
		CFLAGS="$$(LIBUV_CFLAGS_$(1)) $$(SNAP_DEFINES)" \
		LDFLAGS="$$(CFG_GCCISH_LINK_FLAGS_$(1))" \
		CC="$$(CC_$(1))" \
		CXX="$$(CXX_$(1))" \
		AR="$$(AR_$(1))" \
		$$(LIBUV_ARGS_$(1)) \
		BUILDTYPE=Release \
		NO_LOAD="$$(LIBUV_NO_LOAD)" \
		V=$$(VERBOSE)

endif

endef

# Instantiate template for all stages/targets
$(foreach target,$(CFG_TARGET), \
     $(eval $(call DEF_THIRD_PARTY_TARGETS,$(target))))
