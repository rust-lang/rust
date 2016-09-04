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
NATIVE_LIBS := hoedown miniz rust_test_helpers

# A macro to add a generic implementation of intrinsics iff a arch optimized implementation is not
# already in the list.
# $(1) is the target
# $(2) is the intrinsic
define ADD_INTRINSIC
  ifeq ($$(findstring X,$$(foreach intrinsic,$$(COMPRT_OBJS_$(1)),$$(if $$(findstring $(2),$$(intrinsic)),X,))),)
    COMPRT_OBJS_$(1) += $(2)
  endif
endef

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

ifeq ($$(CFG_WINDOWSY_$(1)),1)
  # A bit of history here, this used to be --enable-lazy-lock added in #14006
  # which was filed with jemalloc in jemalloc/jemalloc#83 which was also
  # reported to MinGW: http://sourceforge.net/p/mingw-w64/bugs/395/
  #
  # When updating jemalloc to 4.0, however, it was found that binaries would
  # exit with the status code STATUS_RESOURCE_NOT_OWNED indicating that a thread
  # was unlocking a mutex it never locked. Disabling this "lazy lock" option
  # seems to fix the issue, but it was enabled by default for MinGW targets in
  # 13473c7 for jemalloc.
  #
  # As a result of all that, force disabling lazy lock on Windows, and after
  # reading some code it at least *appears* that the initialization of mutexes
  # is otherwise ok in jemalloc, so shouldn't cause problems hopefully...
  #
  # tl;dr: make windows behave like other platforms by disabling lazy locking,
  #        but requires passing an option due to a historical default with
  #        jemalloc.
  JEMALLOC_ARGS_$(1) := --disable-lazy-lock
else ifeq ($(OSTYPE_$(1)), apple-ios)
  JEMALLOC_ARGS_$(1) := --disable-tls
else ifeq ($(findstring android, $(OSTYPE_$(1))), android)
  # We force android to have prefixed symbols because apparently replacement of
  # the libc allocator doesn't quite work. When this was tested (unprefixed
  # symbols), it was found that the `realpath` function in libc would allocate
  # with libc malloc (not jemalloc malloc), and then the standard library would
  # free with jemalloc free, causing a segfault.
  #
  # If the test suite passes, however, without symbol prefixes then we should be
  # good to go!
  JEMALLOC_ARGS_$(1) := --disable-tls --with-jemalloc-prefix=je_
else ifeq ($(findstring dragonfly, $(OSTYPE_$(1))), dragonfly)
  JEMALLOC_ARGS_$(1) := --with-jemalloc-prefix=je_
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
		$$(JEMALLOC_ARGS_$(1)) $(CFG_JEMALLOC_FLAGS) \
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

# Everything below is a manual compilation of compiler-rt, disregarding its
# build system. See comments in `src/bootstrap/native.rs` for more information.

COMPRT_NAME_$(1) := $$(call CFG_STATIC_LIB_NAME_$(1),compiler-rt)
COMPRT_LIB_$(1) := $$(RT_OUTPUT_DIR_$(1))/$$(COMPRT_NAME_$(1))
COMPRT_BUILD_DIR_$(1) := $$(RT_OUTPUT_DIR_$(1))/compiler-rt

# We must avoid compiling both a generic implementation (e.g. `floatdidf.c) and an arch optimized
# implementation (e.g. `x86_64/floatdidf.S) of the same symbol (e.g. `floatdidf) because that causes
# linker errors. To avoid that, we first add all the arch optimized implementations and then add the
# generic implementations if and only if its arch optimized version is not already in the list. This
# last part is handled by the ADD_INTRINSIC macro.

COMPRT_OBJS_$(1) :=

ifeq ($$(findstring msvc,$(1)),)
ifeq ($$(findstring x86_64,$(1)),x86_64)
COMPRT_OBJS_$(1) += \
      x86_64/chkstk.o \
      x86_64/chkstk2.o \
      x86_64/floatdidf.o \
      x86_64/floatdisf.o \
      x86_64/floatdixf.o \
      x86_64/floatundidf.o \
      x86_64/floatundisf.o \
      x86_64/floatundixf.o
endif

ifeq ($$(findstring i686,$$(patsubts i%86,i686,$(1))),i686)
COMPRT_OBJS_$(1) += \
      i386/ashldi3.o \
      i386/ashrdi3.o \
      i386/chkstk.o \
      i386/chkstk2.o \
      i386/divdi3.o \
      i386/floatdidf.o \
      i386/floatdisf.o \
      i386/floatdixf.o \
      i386/floatundidf.o \
      i386/floatundisf.o \
      i386/floatundixf.o \
      i386/lshrdi3.o \
      i386/moddi3.o \
      i386/muldi3.o \
      i386/udivdi3.o \
      i386/umoddi3.o
endif

else

ifeq ($$(findstring x86_64,$(1)),x86_64)
COMPRT_OBJS_$(1) += \
      x86_64/floatdidf.o \
      x86_64/floatdisf.o \
      x86_64/floatdixf.o
endif

endif

# Generic ARM sources, nothing compiles on iOS though
ifeq ($$(findstring arm,$(1)),arm)
ifeq ($$(findstring ios,$(1)),)
COMPRT_OBJS_$(1) += \
  arm/aeabi_cdcmp.o \
  arm/aeabi_cdcmpeq_check_nan.o \
  arm/aeabi_cfcmp.o \
  arm/aeabi_cfcmpeq_check_nan.o \
  arm/aeabi_dcmp.o \
  arm/aeabi_div0.o \
  arm/aeabi_drsub.o \
  arm/aeabi_fcmp.o \
  arm/aeabi_frsub.o \
  arm/aeabi_idivmod.o \
  arm/aeabi_ldivmod.o \
  arm/aeabi_memcmp.o \
  arm/aeabi_memcpy.o \
  arm/aeabi_memmove.o \
  arm/aeabi_memset.o \
  arm/aeabi_uidivmod.o \
  arm/aeabi_uldivmod.o \
  arm/bswapdi2.o \
  arm/bswapsi2.o \
  arm/clzdi2.o \
  arm/clzsi2.o \
  arm/comparesf2.o \
  arm/divmodsi4.o \
  arm/divsi3.o \
  arm/modsi3.o \
  arm/switch16.o \
  arm/switch32.o \
  arm/switch8.o \
  arm/switchu8.o \
  arm/sync_synchronize.o \
  arm/udivmodsi4.o \
  arm/udivsi3.o \
  arm/umodsi3.o
endif
endif

# Thumb sources
ifeq ($$(findstring armv7,$(1)),armv7)
COMPRT_OBJS_$(1) += \
  arm/sync_fetch_and_add_4.o \
  arm/sync_fetch_and_add_8.o \
  arm/sync_fetch_and_and_4.o \
  arm/sync_fetch_and_and_8.o \
  arm/sync_fetch_and_max_4.o \
  arm/sync_fetch_and_max_8.o \
  arm/sync_fetch_and_min_4.o \
  arm/sync_fetch_and_min_8.o \
  arm/sync_fetch_and_nand_4.o \
  arm/sync_fetch_and_nand_8.o \
  arm/sync_fetch_and_or_4.o \
  arm/sync_fetch_and_or_8.o \
  arm/sync_fetch_and_sub_4.o \
  arm/sync_fetch_and_sub_8.o \
  arm/sync_fetch_and_umax_4.o \
  arm/sync_fetch_and_umax_8.o \
  arm/sync_fetch_and_umin_4.o \
  arm/sync_fetch_and_umin_8.o \
  arm/sync_fetch_and_xor_4.o \
  arm/sync_fetch_and_xor_8.o
endif

# VFP sources
ifeq ($$(findstring eabihf,$(1)),eabihf)
COMPRT_OBJS_$(1) += \
  arm/adddf3vfp.o \
  arm/addsf3vfp.o \
  arm/divdf3vfp.o \
  arm/divsf3vfp.o \
  arm/eqdf2vfp.o \
  arm/eqsf2vfp.o \
  arm/extendsfdf2vfp.o \
  arm/fixdfsivfp.o \
  arm/fixsfsivfp.o \
  arm/fixunsdfsivfp.o \
  arm/fixunssfsivfp.o \
  arm/floatsidfvfp.o \
  arm/floatsisfvfp.o \
  arm/floatunssidfvfp.o \
  arm/floatunssisfvfp.o \
  arm/gedf2vfp.o \
  arm/gesf2vfp.o \
  arm/gtdf2vfp.o \
  arm/gtsf2vfp.o \
  arm/ledf2vfp.o \
  arm/lesf2vfp.o \
  arm/ltdf2vfp.o \
  arm/ltsf2vfp.o \
  arm/muldf3vfp.o \
  arm/mulsf3vfp.o \
  arm/negdf2vfp.o \
  arm/negsf2vfp.o \
  arm/nedf2vfp.o \
  arm/nesf2vfp.o \
  arm/restore_vfp_d8_d15_regs.o \
  arm/save_vfp_d8_d15_regs.o \
  arm/subdf3vfp.o \
  arm/subsf3vfp.o \
  arm/truncdfsf2vfp.o \
  arm/unorddf2vfp.o \
  arm/unordsf2vfp.o
endif

$(foreach intrinsic,absvdi2.o \
  absvsi2.o \
  adddf3.o \
  addsf3.o \
  addvdi3.o \
  addvsi3.o \
  apple_versioning.o \
  ashldi3.o \
  ashrdi3.o \
  clear_cache.o \
  clzdi2.o \
  clzsi2.o \
  cmpdi2.o \
  comparedf2.o \
  comparesf2.o \
  ctzdi2.o \
  ctzsi2.o \
  divdc3.o \
  divdf3.o \
  divdi3.o \
  divmoddi4.o \
  divmodsi4.o \
  divsc3.o \
  divsf3.o \
  divsi3.o \
  divxc3.o \
  extendsfdf2.o \
  extendhfsf2.o \
  ffsdi2.o \
  fixdfdi.o \
  fixdfsi.o \
  fixsfdi.o \
  fixsfsi.o \
  fixunsdfdi.o \
  fixunsdfsi.o \
  fixunssfdi.o \
  fixunssfsi.o \
  fixunsxfdi.o \
  fixunsxfsi.o \
  fixxfdi.o \
  floatdidf.o \
  floatdisf.o \
  floatdixf.o \
  floatsidf.o \
  floatsisf.o \
  floatundidf.o \
  floatundisf.o \
  floatundixf.o \
  floatunsidf.o \
  floatunsisf.o \
  int_util.o \
  lshrdi3.o \
  moddi3.o \
  modsi3.o \
  muldc3.o \
  muldf3.o \
  muldi3.o \
  mulodi4.o \
  mulosi4.o \
  muloti4.o \
  mulsc3.o \
  mulsf3.o \
  mulvdi3.o \
  mulvsi3.o \
  mulxc3.o \
  negdf2.o \
  negdi2.o \
  negsf2.o \
  negvdi2.o \
  negvsi2.o \
  paritydi2.o \
  paritysi2.o \
  popcountdi2.o \
  popcountsi2.o \
  powidf2.o \
  powisf2.o \
  powixf2.o \
  subdf3.o \
  subsf3.o \
  subvdi3.o \
  subvsi3.o \
  truncdfhf2.o \
  truncdfsf2.o \
  truncsfhf2.o \
  ucmpdi2.o \
  udivdi3.o \
  udivmoddi4.o \
  udivmodsi4.o \
  udivsi3.o \
  umoddi3.o \
  umodsi3.o,
  $(call ADD_INTRINSIC,$(1),$(intrinsic)))

ifeq ($$(findstring ios,$(1)),)
$(foreach intrinsic,absvti2.o \
  addtf3.o \
  addvti3.o \
  ashlti3.o \
  ashrti3.o \
  clzti2.o \
  cmpti2.o \
  ctzti2.o \
  divtf3.o \
  divti3.o \
  ffsti2.o \
  fixdfti.o \
  fixsfti.o \
  fixunsdfti.o \
  fixunssfti.o \
  fixunsxfti.o \
  fixxfti.o \
  floattidf.o \
  floattisf.o \
  floattixf.o \
  floatuntidf.o \
  floatuntisf.o \
  floatuntixf.o \
  lshrti3.o \
  modti3.o \
  multf3.o \
  multi3.o \
  mulvti3.o \
  negti2.o \
  negvti2.o \
  parityti2.o \
  popcountti2.o \
  powitf2.o \
  subtf3.o \
  subvti3.o \
  trampoline_setup.o \
  ucmpti2.o \
  udivmodti4.o \
  udivti3.o \
  umodti3.o,
  $(call ADD_INTRINSIC,$(1),$(intrinsic)))
endif

ifeq ($$(findstring apple,$(1)),apple)
$(foreach intrinsic,atomic_flag_clear.o \
  atomic_flag_clear_explicit.o \
  atomic_flag_test_and_set.o \
  atomic_flag_test_and_set_explicit.o \
  atomic_signal_fence.o \
  atomic_thread_fence.o,
  $(call ADD_INTRINSIC,$(1),$(intrinsic)))
endif

ifeq ($$(findstring windows,$(1)),)
$(call ADD_INTRINSIC,$(1),emutls.o)
endif

ifeq ($$(findstring msvc,$(1)),)

ifeq ($$(findstring freebsd,$(1)),)
$(call ADD_INTRINSIC,$(1),gcc_personality_v0.o)
endif
endif

ifeq ($$(findstring aarch64,$(1)),aarch64)
$(foreach intrinsic,comparetf2.o \
  extenddftf2.o \
  extendsftf2.o \
  fixtfdi.o \
  fixtfsi.o \
  fixtfti.o \
  fixunstfdi.o \
  fixunstfsi.o \
  fixunstfti.o \
  floatditf.o \
  floatsitf.o \
  floatunditf.o \
  floatunsitf.o \
  multc3.o \
  trunctfdf2.o \
  trunctfsf2.o,
  $(call ADD_INTRINSIC,$(1),$(intrinsic)))
endif

ifeq ($$(findstring msvc,$(1)),msvc)
$$(COMPRT_BUILD_DIR_$(1))/%.o: CFLAGS += -Zl -D__func__=__FUNCTION__
else
$$(COMPRT_BUILD_DIR_$(1))/%.o: CFLAGS += -fno-builtin -fvisibility=hidden \
	-fomit-frame-pointer -ffreestanding
endif

COMPRT_OBJS_$(1) := $$(COMPRT_OBJS_$(1):%=$$(COMPRT_BUILD_DIR_$(1))/%)

$$(COMPRT_BUILD_DIR_$(1))/%.o: $(S)src/compiler-rt/lib/builtins/%.c
	@mkdir -p $$(@D)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1),$$@,$$<)

$$(COMPRT_BUILD_DIR_$(1))/%.o: $(S)src/compiler-rt/lib/builtins/%.S \
	    $$(LLVM_CONFIG_$$(CFG_BUILD))
	@mkdir -p $$(@D)
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_ASSEMBLE_$(1),$$@,$$<)

ifeq ($$(findstring msvc,$(1)),msvc)
$$(COMPRT_BUILD_DIR_$(1))/%.o: \
	export INCLUDE := $$(CFG_MSVC_INCLUDE_PATH_$$(HOST_$(1)))
endif

ifeq ($$(findstring emscripten,$(1)),emscripten)
# FIXME: emscripten doesn't use compiler-rt and can't build it without
# further hacks
COMPRT_OBJS_$(1) :=
endif

$$(COMPRT_LIB_$(1)): $$(COMPRT_OBJS_$(1))
	@$$(call E, link: $$@)
	$$(Q)$$(call CFG_CREATE_ARCHIVE_$(1),$$@) $$^

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

else ifeq ($$(findstring ios,$$(OSTYPE_$(1))),ios)
# See comment above
$$(BACKTRACE_LIB_$(1)):
	touch $$@
else ifeq ($$(findstring msvc,$(1)),msvc)
# See comment above
$$(BACKTRACE_LIB_$(1)):
	touch $$@
else ifeq ($$(findstring emscripten,$(1)),emscripten)
# FIXME: libbacktrace doesn't understand the emscripten triple
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
	      CFLAGS="$$(CFG_GCCISH_CFLAGS_$(1)) -Wno-error -fno-stack-protector" \
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

endif

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
else
# Ask gcc where it is
$$(RT_OUTPUT_DIR_$(1))/%:
	cp $$(shell $$(CC_$(1)) -print-file-name=$$(@F)) $$@
endif

endef

# Instantiate template for all stages/targets
$(foreach target,$(CFG_TARGET), \
     $(eval $(call DEF_THIRD_PARTY_TARGETS,$(target))))
