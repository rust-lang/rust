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
NATIVE_LIBS := rust_builtin hoedown uv_support morestack miniz context_switch \
		rustrt_native rust_test_helpers

# $(1) is the target triple
define NATIVE_LIBRARIES

NATIVE_DEPS_hoedown_$(1) := hoedown/src/autolink.c \
			hoedown/src/buffer.c \
			hoedown/src/document.c \
			hoedown/src/escape.c \
			hoedown/src/html.c \
			hoedown/src/html_blocks.c \
			hoedown/src/html_smartypants.c \
			hoedown/src/stack.c \
			hoedown/src/version.c
NATIVE_DEPS_uv_support_$(1) := rust_uv.c
NATIVE_DEPS_miniz_$(1) = miniz.c
NATIVE_DEPS_rust_builtin_$(1) := rust_builtin.c \
			rust_android_dummy.c
NATIVE_DEPS_rustrt_native_$(1) := \
			rust_try.ll \
			arch/$$(HOST_$(1))/record_sp.S
NATIVE_DEPS_rust_test_helpers_$(1) := rust_test_helpers.c
NATIVE_DEPS_morestack_$(1) := arch/$$(HOST_$(1))/morestack.S
NATIVE_DEPS_context_switch_$(1) := \
			arch/$$(HOST_$(1))/_context.S

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
		-I $$(S)src/rt/hoedown/src \
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

$(foreach target,$(CFG_TARGET), \
 $(eval $(call RUNTIME_RULES,$(target))))
$(foreach lib,$(NATIVE_LIBS), \
 $(foreach target,$(CFG_TARGET), \
  $(eval $(call THIRD_PARTY_LIB,$(target),$(lib)))))


################################################################################
# Building third-party targets with external build systems
#
# This location is meant for dependencies which have external build systems. It
# is still assumed that the output of each of these steps is a static library
# in the correct location.
################################################################################

################################################################################
# libuv
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
  # This isn't necessarily a desired option, but it's harmless and works around
  # what appears to be a mingw-w64 bug.
  #
  # https://sourceforge.net/p/mingw-w64/bugs/395/
  JEMALLOC_ARGS_$(1) := --enable-lazy-lock
else ifeq ($(OSTYPE_$(1)), apple-darwin)
  LIBUV_OSTYPE_$(1) := mac
else ifeq ($(OSTYPE_$(1)), apple-ios)
  LIBUV_OSTYPE_$(1) := ios
  JEMALLOC_ARGS_$(1) := --disable-tls
else ifeq ($(OSTYPE_$(1)), unknown-freebsd)
  LIBUV_OSTYPE_$(1) := freebsd
else ifeq ($(OSTYPE_$(1)), unknown-dragonfly)
  LIBUV_OSTYPE_$(1) := freebsd
  # required on DragonFly, otherwise gyp fails with a Python exception
  LIBUV_GYP_ARGS_$(1) := --no-parallel
else ifeq ($(OSTYPE_$(1)), linux-androideabi)
  LIBUV_OSTYPE_$(1) := android
  LIBUV_ARGS_$(1) := PLATFORM=android host=android OS=linux
  JEMALLOC_ARGS_$(1) := --disable-tls
else
  LIBUV_OSTYPE_$(1) := linux
endif

LIBUV_NAME_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),uv)
LIBUV_DIR_$(1) := $$(RT_OUTPUT_DIR_$(1))/libuv
LIBUV_LIB_$(1) := $$(RT_OUTPUT_DIR_$(1))/$$(LIBUV_NAME_$(1))

LIBUV_MAKEFILE_$(1) := $$(CFG_BUILD_DIR)$$(RT_OUTPUT_DIR_$(1))/libuv/Makefile
LIBUV_BUILD_DIR_$(1) := $$(CFG_BUILD_DIR)$$(RT_OUTPUT_DIR_$(1))/libuv
LIBUV_XCODEPROJ_$(1) := $$(LIBUV_BUILD_DIR_$(1))/uv.xcodeproj

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
	   -Goutput_dir=$$(@D) $$(LIBUV_GYP_ARGS_$(1)) --generator-output $$(@D))
	touch $$@

# Windows has a completely different build system for libuv because of mingw. In
# theory when we support msvc then we should be using gyp's msvc output instead
# of mingw's makefile for windows
ifdef CFG_WINDOWSY_$(1)
LIBUV_LOCAL_$(1) := $$(S)src/libuv/libuv.a
$$(LIBUV_LOCAL_$(1)): $$(LIBUV_DEPS) $$(MKFILE_DEPS)
	$$(Q)$$(MAKE) -C $$(S)src/libuv -f Makefile.mingw \
		LDFLAGS="$$(CFG_GCCISH_LINK_FLAGS_$(1))" \
		CC="$$(CC_$(1)) $$(LIBUV_CFLAGS_$(1)) $$(SNAP_DEFINES)" \
		CXX="$$(CXX_$(1))" \
		AR="$$(AR_$(1))" \
		V=$$(VERBOSE)
else ifeq ($(OSTYPE_$(1)), apple-ios) # iOS
$$(LIBUV_XCODEPROJ_$(1)): $$(LIBUV_DEPS) $$(MKFILE_DEPS) $$(LIBUV_STAMP_$(1))
	cp -rf $(S)src/libuv/ $$(LIBUV_BUILD_DIR_$(1))
	(cd $$(LIBUV_BUILD_DIR_$(1)) && \
	 $$(CFG_PYTHON) ./gyp_uv.py -f xcode \
	   -D ninja \
	   -R libuv)
	touch $$@

LIBUV_XCODE_OUT_LIB_$(1) := $$(LIBUV_BUILD_DIR_$(1))/build/Release-$$(CFG_SDK_NAME_$(1))/libuv.a

$$(LIBUV_LIB_$(1)): $$(LIBUV_XCODE_OUT_LIB_$(1)) $$(MKFILE_DEPS)
	$$(Q)cp $$< $$@
$$(LIBUV_XCODE_OUT_LIB_$(1)): $$(LIBUV_DEPS) $$(LIBUV_XCODEPROJ_$(1)) \
				    $$(MKFILE_DEPS)
	$$(Q)xcodebuild -project $$(LIBUV_BUILD_DIR_$(1))/uv.xcodeproj \
		CFLAGS="$$(LIBUV_CFLAGS_$(1)) $$(SNAP_DEFINES)" \
		LDFLAGS="$$(CFG_GCCISH_LINK_FLAGS_$(1))" \
		$$(LIBUV_ARGS_$(1)) \
		V=$$(VERBOSE) \
		-configuration Release \
		-sdk "$$(CFG_SDK_NAME_$(1))" \
		ARCHS="$$(CFG_SDK_ARCHS_$(1))"
	$$(Q)touch $$@
else
LIBUV_LOCAL_$(1) := $$(LIBUV_DIR_$(1))/Release/libuv.a
$$(LIBUV_LOCAL_$(1)): $$(LIBUV_DEPS) $$(LIBUV_MAKEFILE_$(1)) $$(MKFILE_DEPS)
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
	$$(Q)touch $$@
endif

ifeq ($(1),$$(CFG_BUILD))
ifneq ($$(CFG_LIBUV_ROOT),)
$$(LIBUV_LIB_$(1)): $$(CFG_LIBUV_ROOT)/libuv.a
	$$(Q)cp $$< $$@
else
$$(LIBUV_LIB_$(1)): $$(LIBUV_LOCAL_$(1))
	$$(Q)cp $$< $$@
endif
else
$$(LIBUV_LIB_$(1)): $$(LIBUV_LOCAL_$(1))
	$$(Q)cp $$< $$@
endif

################################################################################
# jemalloc
################################################################################

ifdef CFG_ENABLE_FAST_MAKE
JEMALLOC_DEPS := $(S)/.gitmodules
else
JEMALLOC_DEPS := $(wildcard \
		   $(S)src/jemalloc/* \
		   $(S)src/jemalloc/*/* \
		   $(S)src/jemalloc/*/*/* \
		   $(S)src/jemalloc/*/*/*/*)
endif

JEMALLOC_NAME_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),jemalloc)
ifeq ($$(CFG_WINDOWSY_$(1)),1)
  JEMALLOC_REAL_NAME_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),jemalloc_s)
else
  JEMALLOC_REAL_NAME_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),jemalloc_pic)
endif
JEMALLOC_LIB_$(1) := $$(RT_OUTPUT_DIR_$(1))/$$(JEMALLOC_NAME_$(1))
JEMALLOC_BUILD_DIR_$(1) := $$(RT_OUTPUT_DIR_$(1))/jemalloc
JEMALLOC_LOCAL_$(1) := $$(JEMALLOC_BUILD_DIR_$(1))/lib/$$(JEMALLOC_REAL_NAME_$(1))

$$(JEMALLOC_LOCAL_$(1)): $$(JEMALLOC_DEPS) $$(MKFILE_DEPS)
	@$$(call E, make: jemalloc)
	cd "$$(JEMALLOC_BUILD_DIR_$(1))"; "$(S)src/jemalloc/configure" \
		$$(JEMALLOC_ARGS_$(1)) --with-jemalloc-prefix=je_ \
		--build=$(CFG_BUILD) --host=$(1) \
		CC="$$(CC_$(1))" \
		AR="$$(AR_$(1))" \
		RANLIB="$$(AR_$(1)) s" \
		CPPFLAGS="-I $(S)src/rt/" \
		EXTRA_CFLAGS="$$(CFG_CFLAGS_$(1)) $$(CFG_JEMALLOC_CFLAGS_$(1)) -g1"
	$$(Q)$$(MAKE) -C "$$(JEMALLOC_BUILD_DIR_$(1))" build_lib_static

ifeq ($$(CFG_DISABLE_JEMALLOC),)
RUSTFLAGS_alloc := --cfg jemalloc
ifeq ($(1),$$(CFG_BUILD))
ifneq ($$(CFG_JEMALLOC_ROOT),)
$$(JEMALLOC_LIB_$(1)): $$(CFG_JEMALLOC_ROOT)/libjemalloc_pic.a
	@$$(call E, copy: jemalloc)
	$$(Q)cp $$< $$@
else
$$(JEMALLOC_LIB_$(1)): $$(JEMALLOC_LOCAL_$(1))
	$$(Q)cp $$< $$@
endif
else
$$(JEMALLOC_LIB_$(1)): $$(JEMALLOC_LOCAL_$(1))
	$$(Q)cp $$< $$@
endif
else
$$(JEMALLOC_LIB_$(1)): $$(MKFILE_DEPS)
	$$(Q)touch $$@
endif

################################################################################
# compiler-rt
################################################################################

ifdef CFG_ENABLE_FAST_MAKE
COMPRT_DEPS := $(S)/.gitmodules
else
COMPRT_DEPS := $(wildcard \
              $(S)src/compiler-rt/* \
              $(S)src/compiler-rt/*/* \
              $(S)src/compiler-rt/*/*/* \
              $(S)src/compiler-rt/*/*/*/*)
endif

COMPRT_NAME_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),compiler-rt)
COMPRT_LIB_$(1) := $$(RT_OUTPUT_DIR_$(1))/$$(COMPRT_NAME_$(1))
COMPRT_BUILD_DIR_$(1) := $$(RT_OUTPUT_DIR_$(1))/compiler-rt

$$(COMPRT_LIB_$(1)): $$(COMPRT_DEPS) $$(MKFILE_DEPS)
	@$$(call E, make: compiler-rt)
	$$(Q)$$(MAKE) -C "$(S)src/compiler-rt" \
		ProjSrcRoot="$(S)src/compiler-rt" \
		ProjObjRoot="$$(abspath $$(COMPRT_BUILD_DIR_$(1)))" \
		CC="$$(CC_$(1))" \
		AR="$$(AR_$(1))" \
		RANLIB="$$(AR_$(1)) s" \
		CFLAGS="$$(CFG_GCCISH_CFLAGS_$(1))" \
		TargetTriple=$(1) \
		triple-builtins
	$$(Q)cp $$(COMPRT_BUILD_DIR_$(1))/triple/builtins/libcompiler_rt.a $$(COMPRT_LIB_$(1))

################################################################################
# libbacktrace
#
# We use libbacktrace on linux to get symbols in backtraces, but only on linux.
# Elsewhere we use other system utilities, so this library is only built on
# linux.
################################################################################

BACKTRACE_NAME_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),backtrace)
BACKTRACE_LIB_$(1) := $$(RT_OUTPUT_DIR_$(1))/$$(BACKTRACE_NAME_$(1))
BACKTRACE_BUILD_DIR_$(1) := $$(RT_OUTPUT_DIR_$(1))/libbacktrace

# We don't use this on platforms that aren't linux-based, so just make the file
# available, the compilation of libstd won't actually build it.
ifeq ($$(findstring darwin,$$(OSTYPE_$(1))),darwin)
# See comment above
$$(BACKTRACE_LIB_$(1)):
	touch $$@

else
ifeq ($$(findstring ios,$$(OSTYPE_$(1))),ios)
# See comment above
$$(BACKTRACE_LIB_$(1)):
	touch $$@
else

ifeq ($$(CFG_WINDOWSY_$(1)),1)
# See comment above
$$(BACKTRACE_LIB_$(1)):
	touch $$@
else

ifdef CFG_ENABLE_FAST_MAKE
BACKTRACE_DEPS := $(S)/.gitmodules
else
BACKTRACE_DEPS := $(wildcard $(S)src/libbacktrace/*)
endif

# We need to export CFLAGS because otherwise it doesn't pick up cross compile
# builds. If libbacktrace doesn't realize this, it will attempt to read 64-bit
# elf headers when compiled for a 32-bit system, yielding blank backtraces.
#
# This also removes the -Werror flag specifically to prevent errors during
# configuration.
#
# Down below you'll also see echos into the config.h generated by the
# ./configure script. This is done to force libbacktrace to *not* use the
# atomic/sync functionality because it pulls in unnecessary dependencies and we
# never use it anyway.
$$(BACKTRACE_BUILD_DIR_$(1))/Makefile: \
		export CFLAGS:=$$(CFG_GCCISH_CFLAGS_$(1):-Werror=) \
				-fno-stack-protector
$$(BACKTRACE_BUILD_DIR_$(1))/Makefile: export CC:=$$(CC_$(1))
$$(BACKTRACE_BUILD_DIR_$(1))/Makefile: export AR:=$$(AR_$(1))
$$(BACKTRACE_BUILD_DIR_$(1))/Makefile: export RANLIB:=$$(AR_$(1)) s
$$(BACKTRACE_BUILD_DIR_$(1))/Makefile: $$(BACKTRACE_DEPS) $$(MKFILE_DEPS)
	$$(Q)rm -rf $$(BACKTRACE_BUILD_DIR_$(1))
	$$(Q)mkdir -p $$(BACKTRACE_BUILD_DIR_$(1))
	$$(Q)(cd $$(BACKTRACE_BUILD_DIR_$(1)) && \
	      $(S)src/libbacktrace/configure --target=$(1) --host=$(CFG_BUILD))
	$$(Q)echo '#undef HAVE_ATOMIC_FUNCTIONS' >> \
	      $$(BACKTRACE_BUILD_DIR_$(1))/config.h
	$$(Q)echo '#undef HAVE_SYNC_FUNCTIONS' >> \
	      $$(BACKTRACE_BUILD_DIR_$(1))/config.h

$$(BACKTRACE_LIB_$(1)): $$(BACKTRACE_BUILD_DIR_$(1))/Makefile $$(MKFILE_DEPS)
	@$$(call E, make: libbacktrace)
	$$(Q)$$(MAKE) -C $$(BACKTRACE_BUILD_DIR_$(1)) \
		INCDIR=$(S)src/libbacktrace
	$$(Q)cp $$(BACKTRACE_BUILD_DIR_$(1))/.libs/libbacktrace.a $$@

endif # endif for windowsy
endif # endif for ios
endif # endif for darwin

endef

# Instantiate template for all stages/targets
$(foreach target,$(CFG_TARGET), \
     $(eval $(call DEF_THIRD_PARTY_TARGETS,$(target))))
