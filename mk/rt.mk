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
NATIVE_LIBS := rust_builtin hoedown miniz rust_test_helpers

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
NATIVE_DEPS_miniz_$(1) = miniz.c
NATIVE_DEPS_rust_builtin_$(1) := rust_builtin.c \
			rust_android_dummy.c
NATIVE_DEPS_rust_test_helpers_$(1) := rust_test_helpers.c

################################################################################
# You shouldn't find it that necessary to edit anything below this line.
################################################################################

# While we're defining the native libraries for each target, we define some
# common rules used to build files for various targets.

RT_OUTPUT_DIR_$(1) := $(1)/rt

$$(RT_OUTPUT_DIR_$(1))/%.o: $(S)src/rt/%.c $$(MKFILE_DEPS)
	@mkdir -p $$(@D)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, \
		$$(call CFG_CC_INCLUDE_$(1),$$(S)src/rt/hoedown/src) \
		$$(call CFG_CC_INCLUDE_$(1),$$(S)src/rt) \
                 $$(RUNTIME_CFLAGS_$(1))) $$<

$$(RT_OUTPUT_DIR_$(1))/%.o: $(S)src/rt/%.S $$(MKFILE_DEPS) \
	    $$(LLVM_CONFIG_$$(CFG_BUILD))
	@mkdir -p $$(@D)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_ASSEMBLE_$(1),$$@,$$<)

# On MSVC targets the compiler's default include path (e.g. where to find system
# headers) is specified by the INCLUDE environment variable. This may not be set
# so the ./configure script scraped the relevant values and this is the location
# that we put them into cl.exe's environment.
ifeq ($$(findstring msvc,$(1)),msvc)
$$(RT_OUTPUT_DIR_$(1))/%.o: \
	export INCLUDE := $$(CFG_MSVC_INCLUDE_PATH_$$(HOST_$(1)))
$(1)/rustllvm/%.o: \
	export INCLUDE := $$(CFG_MSVC_INCLUDE_PATH_$$(HOST_$(1)))
endif
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
OBJS_$(2)_$(1) := $$(OBJS_$(2)_$(1):.S=.o)
NATIVE_$(2)_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),$(2))
$$(RT_OUTPUT_DIR_$(1))/$$(NATIVE_$(2)_$(1)): $$(OBJS_$(2)_$(1))
	@$$(call E, link: $$@)
	$$(Q)$$(call CFG_CREATE_ARCHIVE_$(1),$$@) $$^

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

define DEF_THIRD_PARTY_TARGETS

# $(1) is the target triple

ifeq ($$(CFG_WINDOWSY_$(1)), 1)
  # This isn't necessarily a desired option, but it's harmless and works around
  # what appears to be a mingw-w64 bug.
  #
  # https://sourceforge.net/p/mingw-w64/bugs/395/
  JEMALLOC_ARGS_$(1) := --enable-lazy-lock
else ifeq ($(OSTYPE_$(1)), apple-ios)
  JEMALLOC_ARGS_$(1) := --disable-tls
else ifeq ($(findstring android, $(OSTYPE_$(1))), android)
  JEMALLOC_ARGS_$(1) := --disable-tls
endif

ifdef CFG_ENABLE_DEBUG_JEMALLOC
  JEMALLOC_ARGS_$(1) += --enable-debug --enable-fill
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

# See #17183 for details, this file is touched during the build process so we
# don't want to consider it as a dependency.
JEMALLOC_DEPS := $(filter-out $(S)src/jemalloc/VERSION,$(JEMALLOC_DEPS))

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
		$$(JEMALLOC_ARGS_$(1)) --with-jemalloc-prefix=je_ $(CFG_JEMALLOC_FLAGS) \
		--build=$$(CFG_GNU_TRIPLE_$(CFG_BUILD)) --host=$$(CFG_GNU_TRIPLE_$(1)) \
		CC="$$(CC_$(1)) $$(CFG_JEMALLOC_CFLAGS_$(1))" \
		AR="$$(AR_$(1))" \
		RANLIB="$$(AR_$(1)) s" \
		CPPFLAGS="-I $(S)src/rt/" \
		EXTRA_CFLAGS="-g1 -ffunction-sections -fdata-sections"
	$$(Q)$$(MAKE) -C "$$(JEMALLOC_BUILD_DIR_$(1))" build_lib_static

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

################################################################################
# compiler-rt
################################################################################
COMPRT_NAME_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),compiler-rt)
COMPRT_LIB_$(1) := $$(RT_OUTPUT_DIR_$(1))/$$(COMPRT_NAME_$(1))
COMPRT_BUILD_DIR_$(1) := $$(RT_OUTPUT_DIR_$(1))/compiler-rt
ifeq ($$(findstring msvc,$(1)),msvc)
$$(COMPRT_BUILD_DIR_$(1))/powisf2.obj: $(S)src/compiler-rt/lib/builtins/powisf2.c
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, \
		$$(RUNTIME_CFLAGS_$(1))) $$<
$$(COMPRT_BUILD_DIR_$(1))/powidf2.obj: $(S)src/compiler-rt/lib/builtins/powidf2.c
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, \
		$$(RUNTIME_CFLAGS_$(1))) $$<
$$(COMPRT_BUILD_DIR_$(1))/mulodi4.obj: $(S)src/compiler-rt/lib/builtins/mulodi4.c
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, \
		$$(RUNTIME_CFLAGS_$(1))) $$<
$$(COMPRT_LIB_$(1)): $$(COMPRT_BUILD_DIR_$(1))/powisf2.obj $$(COMPRT_BUILD_DIR_$(1))/powidf2.obj $$(COMPRT_BUILD_DIR_$(1))/mulodi4.obj
	@$$(call E, link: $$@)
	$$(Q)$$(call CFG_CREATE_ARCHIVE_$(1),$$@) $$^
else
ifdef CFG_ENABLE_FAST_MAKE
COMPRT_DEPS := $(S)/.gitmodules
else
COMPRT_DEPS := $(wildcard \
              $(S)src/compiler-rt/* \
              $(S)src/compiler-rt/*/* \
              $(S)src/compiler-rt/*/*/* \
              $(S)src/compiler-rt/*/*/*/*)
endif

COMPRT_CC_$(1) := $$(CC_$(1))
COMPRT_AR_$(1) := $$(AR_$(1))
COMPRT_CFLAGS_$(1) := $$(CFG_GCCISH_CFLAGS_$(1))

$$(COMPRT_LIB_$(1)): $$(COMPRT_DEPS) $$(MKFILE_DEPS)
	@$$(call E, make: compiler-rt)
	$$(Q)$$(MAKE) -C "$(S)src/compiler-rt" \
		ProjSrcRoot="$(S)src/compiler-rt" \
		ProjObjRoot="$$(abspath $$(COMPRT_BUILD_DIR_$(1)))" \
		CC='$$(COMPRT_CC_$(1))' \
		AR='$$(COMPRT_AR_$(1))' \
		RANLIB='$$(COMPRT_AR_$(1)) s' \
		CFLAGS="$$(COMPRT_CFLAGS_$(1))" \
		TargetTriple=$(1) \
		triple-builtins
	$$(Q)cp $$(COMPRT_BUILD_DIR_$(1))/triple/builtins/libcompiler_rt.a $$@
endif # msvc
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

# We don't use this on platforms that aren't linux-based (with the exception of
# msys2/mingw builds on windows, which use it to read the dwarf debug
# information) so just make the file available, the compilation of libstd won't
# actually build it.
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

ifeq ($$(findstring msvc,$(1)),msvc)
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
#
# We also use `env PWD=` to clear the PWD environment variable, and then
# execute the command in a new shell. This is necessary to workaround a
# buildbot/msys2 bug: the shell is launched with PWD set to a windows-style path,
# which results in all further uses of `pwd` also printing a windows-style path,
# which breaks libbacktrace's configure script. Clearing PWD within the same
# shell is not sufficient.

$$(BACKTRACE_BUILD_DIR_$(1))/Makefile: $$(BACKTRACE_DEPS) $$(MKFILE_DEPS)
	@$$(call E, configure: libbacktrace for $(1))
	$$(Q)rm -rf $$(BACKTRACE_BUILD_DIR_$(1))
	$$(Q)mkdir -p $$(BACKTRACE_BUILD_DIR_$(1))
	$$(Q)(cd $$(BACKTRACE_BUILD_DIR_$(1)) && env \
	      PWD= \
	      CC="$$(CC_$(1))" \
	      AR="$$(AR_$(1))" \
	      RANLIB="$$(AR_$(1)) s" \
	      CFLAGS="$$(CFG_GCCISH_CFLAGS_$(1):-Werror=) -fno-stack-protector" \
	      $(S)src/libbacktrace/configure --build=$(CFG_GNU_TRIPLE_$(CFG_BUILD)) --host=$(CFG_GNU_TRIPLE_$(1)))
	$$(Q)echo '#undef HAVE_ATOMIC_FUNCTIONS' >> \
	      $$(BACKTRACE_BUILD_DIR_$(1))/config.h
	$$(Q)echo '#undef HAVE_SYNC_FUNCTIONS' >> \
	      $$(BACKTRACE_BUILD_DIR_$(1))/config.h

$$(BACKTRACE_LIB_$(1)): $$(BACKTRACE_BUILD_DIR_$(1))/Makefile $$(MKFILE_DEPS)
	@$$(call E, make: libbacktrace)
	$$(Q)$$(MAKE) -C $$(BACKTRACE_BUILD_DIR_$(1)) \
		INCDIR=$(S)src/libbacktrace
	$$(Q)cp $$(BACKTRACE_BUILD_DIR_$(1))/.libs/libbacktrace.a $$@

endif # endif for msvc
endif # endif for ios
endif # endif for darwin

################################################################################
# libc/libunwind for musl
#
# When we're building a musl-like target we're going to link libc/libunwind
# statically into the standard library and liblibc, so we need to make sure
# they're in a location that we can find
################################################################################

ifeq ($$(findstring musl,$(1)),musl)
$$(RT_OUTPUT_DIR_$(1))/%: $$(CFG_MUSL_ROOT)/lib/%
	cp $$^ $$@
endif

endef

# Instantiate template for all stages/targets
$(foreach target,$(CFG_TARGET), \
     $(eval $(call DEF_THIRD_PARTY_TARGETS,$(target))))
