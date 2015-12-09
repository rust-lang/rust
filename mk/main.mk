# Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

######################################################################
# Version numbers and strings
######################################################################

# The version number
CFG_RELEASE_NUM=1.7.0

# An optional number to put after the label, e.g. '.2' -> '-beta.2'
# NB Make sure it starts with a dot to conform to semver pre-release
# versions (section 9)
CFG_PRERELEASE_VERSION=.1

# Append a version-dependent hash to each library, so we can install different
# versions in the same place
CFG_FILENAME_EXTRA=$(shell printf '%s' $(CFG_RELEASE) | $(CFG_HASH_COMMAND))

ifeq ($(CFG_RELEASE_CHANNEL),stable)
# This is the normal semver version string, e.g. "0.12.0", "0.12.0-nightly"
CFG_RELEASE=$(CFG_RELEASE_NUM)
# This is the string used in dist artifact file names, e.g. "0.12.0", "nightly"
CFG_PACKAGE_VERS=$(CFG_RELEASE_NUM)
CFG_DISABLE_UNSTABLE_FEATURES=1
endif
ifeq ($(CFG_RELEASE_CHANNEL),beta)
CFG_RELEASE=$(CFG_RELEASE_NUM)-beta$(CFG_PRERELEASE_VERSION)
# When building beta distributables just reuse the same "beta" name
# so when we upload we'll always override the previous beta. This
# doesn't actually impact the version reported by rustc - it's just
# for file naming.
CFG_PACKAGE_VERS=beta
CFG_DISABLE_UNSTABLE_FEATURES=1
endif
ifeq ($(CFG_RELEASE_CHANNEL),nightly)
CFG_RELEASE=$(CFG_RELEASE_NUM)-nightly
# When building nightly distributables just reuse the same "nightly" name
# so when we upload we'll always override the previous nighly. This
# doesn't actually impact the version reported by rustc - it's just
# for file naming.
CFG_PACKAGE_VERS=nightly
endif
ifeq ($(CFG_RELEASE_CHANNEL),dev)
CFG_RELEASE=$(CFG_RELEASE_NUM)-dev
CFG_PACKAGE_VERS=$(CFG_RELEASE_NUM)-dev
endif

# The name of the package to use for creating tarballs, installers etc.
CFG_PACKAGE_NAME=rustc-$(CFG_PACKAGE_VERS)

# The version string plus commit information - this is what rustc reports
CFG_VERSION = $(CFG_RELEASE)
CFG_GIT_DIR := $(CFG_SRC_DIR).git
# since $(CFG_GIT) may contain spaces (especially on Windows),
# we need to escape them. (" " to r"\ ")
# Note that $(subst ...) ignores space after `subst`,
# so we use a hack: define $(SPACE) which contains space character.
SPACE :=
SPACE +=
ifneq ($(CFG_GIT),)
ifneq ($(wildcard $(subst $(SPACE),\$(SPACE),$(CFG_GIT_DIR))),)
    CFG_VER_DATE = $(shell git --git-dir='$(CFG_GIT_DIR)' log -1 --date=short --pretty=format:'%cd')
    CFG_VER_HASH = $(shell git --git-dir='$(CFG_GIT_DIR)' rev-parse HEAD)
    CFG_SHORT_VER_HASH = $(shell git --git-dir='$(CFG_GIT_DIR)' rev-parse --short=9 HEAD)
    CFG_VERSION += ($(CFG_SHORT_VER_HASH) $(CFG_VER_DATE))
endif
endif

# Windows exe's need numeric versions - don't use anything but
# numbers and dots here
CFG_VERSION_WIN = $(CFG_RELEASE_NUM)

CFG_INFO := $(info cfg: version $(CFG_VERSION))

######################################################################
# More configuration
######################################################################

MKFILE_DEPS := config.stamp $(call rwildcard,$(CFG_SRC_DIR)mk/,*)
MKFILES_FOR_TARBALL:=$(MKFILE_DEPS)
ifneq ($(NO_MKFILE_DEPS),)
MKFILE_DEPS :=
endif
NON_BUILD_HOST = $(filter-out $(CFG_BUILD),$(CFG_HOST))
NON_BUILD_TARGET = $(filter-out $(CFG_BUILD),$(CFG_TARGET))

ifneq ($(MAKE_RESTARTS),)
CFG_INFO := $(info cfg: make restarts: $(MAKE_RESTARTS))
endif

CFG_INFO := $(info cfg: build triple $(CFG_BUILD))
CFG_INFO := $(info cfg: host triples $(CFG_HOST))
CFG_INFO := $(info cfg: target triples $(CFG_TARGET))

ifneq ($(wildcard $(NON_BUILD_HOST)),)
CFG_INFO := $(info cfg: non-build host triples $(NON_BUILD_HOST))
endif
ifneq ($(wildcard $(NON_BUILD_TARGET)),)
CFG_INFO := $(info cfg: non-build target triples $(NON_BUILD_TARGET))
endif

CFG_RUSTC_FLAGS := $(RUSTFLAGS)
CFG_GCCISH_CFLAGS :=
CFG_GCCISH_LINK_FLAGS :=

CFG_JEMALLOC_FLAGS :=

ifdef CFG_DISABLE_OPTIMIZE
  $(info cfg: disabling rustc optimization (CFG_DISABLE_OPTIMIZE))
  CFG_RUSTC_FLAGS +=
  CFG_JEMALLOC_FLAGS += --enable-debug
else
  # The rtopt cfg turns off runtime sanity checks
  CFG_RUSTC_FLAGS += -O --cfg rtopt
endif

CFG_JEMALLOC_FLAGS += $(JEMALLOC_FLAGS)

ifdef CFG_ENABLE_DEBUG_ASSERTIONS
  $(info cfg: enabling debug assertions (CFG_ENABLE_DEBUG_ASSERTIONS))
  CFG_RUSTC_FLAGS += -C debug-assertions=on
endif

ifdef CFG_ENABLE_DEBUGINFO
  $(info cfg: enabling debuginfo (CFG_ENABLE_DEBUGINFO))
  # FIXME: Re-enable -g in stage0 after new snapshot
  #CFG_RUSTC_FLAGS += -g
  RUSTFLAGS_STAGE1 += -g
  RUSTFLAGS_STAGE2 += -g
  RUSTFLAGS_STAGE3 += -g
endif

ifdef SAVE_TEMPS
  CFG_RUSTC_FLAGS += --save-temps
endif
ifdef ASM_COMMENTS
  CFG_RUSTC_FLAGS += -Z asm-comments
endif
ifdef TIME_PASSES
  CFG_RUSTC_FLAGS += -Z time-passes
endif
ifdef TIME_LLVM_PASSES
  CFG_RUSTC_FLAGS += -Z time-llvm-passes
endif
ifdef TRACE
  CFG_RUSTC_FLAGS += -Z trace
endif
ifdef CFG_ENABLE_RPATH
CFG_RUSTC_FLAGS += -C rpath
endif

# The executables crated during this compilation process have no need to include
# static copies of libstd and libextra. We also generate dynamic versions of all
# libraries, so in the interest of space, prefer dynamic linking throughout the
# compilation process.
#
# Note though that these flags are omitted for the *bins* in stage2+. This means
# that the snapshot will be generated with a statically linked rustc so we only
# have to worry about the distribution of one file (with its native dynamic
# dependencies)
RUSTFLAGS_STAGE0 += -C prefer-dynamic -C no-stack-check
RUSTFLAGS_STAGE1 += -C prefer-dynamic
RUST_LIB_FLAGS_ST2 += -C prefer-dynamic
RUST_LIB_FLAGS_ST3 += -C prefer-dynamic

# Landing pads require a lot of codegen. We can get through bootstrapping faster
# by not emitting them.

ifdef CFG_DISABLE_STAGE0_LANDING_PADS
  RUSTFLAGS_STAGE0 += -Z no-landing-pads
endif

# platform-specific auto-configuration
include $(CFG_SRC_DIR)mk/platform.mk

# Run the stage1/2 compilers under valgrind
ifdef VALGRIND_COMPILE
  CFG_VALGRIND_COMPILE := $(CFG_VALGRIND)
else
  CFG_VALGRIND_COMPILE :=
endif


ifndef CFG_DISABLE_VALGRIND_RPASS
  $(info cfg: enabling valgrind run-pass tests (CFG_ENABLE_VALGRIND_RPASS))
  $(info cfg: valgrind-rpass command set to $(CFG_VALGRIND))
  CFG_VALGRIND_RPASS :=$(CFG_VALGRIND)
else
  $(info cfg: disabling valgrind run-pass tests)
  CFG_VALGRIND_RPASS :=
endif


ifdef CFG_ENABLE_VALGRIND
  $(info cfg: enabling valgrind (CFG_ENABLE_VALGRIND))
  CFG_JEMALLOC_FLAGS += --enable-valgrind
else
  CFG_VALGRIND :=
endif

######################################################################
# Target-and-rule "utility variables"
######################################################################

define DEF_FOR_TARGET
X_$(1) := $(CFG_EXE_SUFFIX_$(1))
ifndef CFG_LLVM_TARGET_$(1)
CFG_LLVM_TARGET_$(1) := $(1)
endif
endef
$(foreach target,$(CFG_TARGET), \
  $(eval $(call DEF_FOR_TARGET,$(target))))

# "Source" files we generate in builddir along the way.
GENERATED :=

# Delete the built-in rules.
.SUFFIXES:
%:: %,v
%:: RCS/%,v
%:: RCS/%
%:: s.%
%:: SCCS/s.%


######################################################################
# Cleaning out old crates
######################################################################

# $(1) is the path for directory to match against
# $(2) is the glob to use in the match
#
# Note that a common bug is to accidentally construct the glob denoted
# by $(2) with a space character prefix, which invalidates the
# construction $(1)$(2).
define CHECK_FOR_OLD_GLOB_MATCHES
  $(Q)MATCHES="$(wildcard $(1))"; if [ -n "$$MATCHES" ] ; then echo "warning: there are previous" \'$(notdir $(2))\' "libraries:" $$MATCHES; fi
endef

# Same interface as above, but deletes rather than just listing the files.
ifdef VERBOSE
define REMOVE_ALL_OLD_GLOB_MATCHES
  $(Q)MATCHES="$(wildcard $(1))"; if [ -n "$$MATCHES" ] ; then echo "warning: removing previous" \'$(notdir $(1))\' "libraries:" $$MATCHES; rm $$MATCHES ; fi
endef
else
define REMOVE_ALL_OLD_GLOB_MATCHES
  $(Q)MATCHES="$(wildcard $(1))"; if [ -n "$$MATCHES" ] ; then rm $$MATCHES ; fi
endef
endif

# We use a different strategy for LIST_ALL_OLD_GLOB_MATCHES_EXCEPT
# than in the macros above because it needs the result of running the
# `ls` command after other rules in the command list have run; the
# macro-expander for $(wildcard ...) would deliver its results too
# soon. (This is in contrast to the macros above, which are meant to
# be run at the outset of a command list in a rule.)
ifdef VERBOSE
define LIST_ALL_OLD_GLOB_MATCHES
  @echo "info: now are following matches for" '$(notdir $(1))' "libraries:"
  @( ls $(1) 2>/dev/null || true )
endef
else
define LIST_ALL_OLD_GLOB_MATCHES
endef
endif

######################################################################
# LLVM macros
######################################################################

LLVM_COMPONENTS=x86 arm aarch64 mips powerpc ipo bitreader bitwriter linker asmparser mcjit \
                interpreter instrumentation

# Only build these LLVM tools
LLVM_TOOLS=bugpoint llc llvm-ar llvm-as llvm-dis llvm-mc opt llvm-extract

define DEF_LLVM_VARS
# The configure script defines these variables with the target triples
# separated by Z. This defines new ones with the expected format.
ifeq ($$(CFG_LLVM_ROOT),)
CFG_LLVM_BUILD_DIR_$(1):=$$(CFG_LLVM_BUILD_DIR_$(subst -,_,$(1)))
CFG_LLVM_INST_DIR_$(1):=$$(CFG_LLVM_INST_DIR_$(subst -,_,$(1)))
else
CFG_LLVM_INST_DIR_$(1):=$$(CFG_LLVM_ROOT)
endif

# Any rules that depend on LLVM should depend on LLVM_CONFIG
LLVM_CONFIG_$(1):=$$(CFG_LLVM_INST_DIR_$(1))/bin/llvm-config$$(X_$(1))
LLVM_MC_$(1):=$$(CFG_LLVM_INST_DIR_$(1))/bin/llvm-mc$$(X_$(1))
LLVM_AR_$(1):=$$(CFG_LLVM_INST_DIR_$(1))/bin/llvm-ar$$(X_$(1))
LLVM_VERSION_$(1)=$$(shell "$$(LLVM_CONFIG_$(1))" --version)
LLVM_BINDIR_$(1)=$$(shell "$$(LLVM_CONFIG_$(1))" --bindir)
LLVM_INCDIR_$(1)=$$(shell "$$(LLVM_CONFIG_$(1))" --includedir)
LLVM_LIBDIR_$(1)=$$(shell "$$(LLVM_CONFIG_$(1))" --libdir)
LLVM_LIBDIR_RUSTFLAGS_$(1)=-L native="$$(LLVM_LIBDIR_$(1))"
LLVM_LDFLAGS_$(1)=$$(shell "$$(LLVM_CONFIG_$(1))" --ldflags)
ifeq ($$(findstring freebsd,$(1)),freebsd)
# On FreeBSD, it may search wrong headers (that are for pre-installed LLVM),
# so we replace -I with -iquote to ensure that it searches bundled LLVM first.
LLVM_CXXFLAGS_$(1)=$$(subst -I, -iquote , $$(shell "$$(LLVM_CONFIG_$(1))" --cxxflags))
else
LLVM_CXXFLAGS_$(1)=$$(shell "$$(LLVM_CONFIG_$(1))" --cxxflags)
endif
LLVM_HOST_TRIPLE_$(1)=$$(shell "$$(LLVM_CONFIG_$(1))" --host-target)

LLVM_AS_$(1)=$$(CFG_LLVM_INST_DIR_$(1))/bin/llvm-as$$(X_$(1))
LLC_$(1)=$$(CFG_LLVM_INST_DIR_$(1))/bin/llc$$(X_$(1))

endef

$(foreach host,$(CFG_HOST), \
 $(eval $(call DEF_LLVM_VARS,$(host))))

######################################################################
# Exports for sub-utilities
######################################################################

# Note that any variable that re-configure should pick up needs to be
# exported

export CFG_SRC_DIR
export CFG_SRC_DIR_RELATIVE
export CFG_BUILD_DIR
ifdef CFG_VER_DATE
export CFG_VER_DATE
endif
ifdef CFG_VER_HASH
export CFG_VER_HASH
endif
export CFG_VERSION
export CFG_VERSION_WIN
export CFG_RELEASE
export CFG_PACKAGE_NAME
export CFG_BUILD
export CFG_RELEASE_CHANNEL
export CFG_LLVM_ROOT
export CFG_PREFIX
export CFG_LIBDIR
export CFG_LIBDIR_RELATIVE
export CFG_DISABLE_INJECT_STD_VERSION
ifdef CFG_DISABLE_UNSTABLE_FEATURES
CFG_INFO := $(info cfg: disabling unstable features (CFG_DISABLE_UNSTABLE_FEATURES))
# Turn on feature-staging
export CFG_DISABLE_UNSTABLE_FEATURES
# Subvert unstable feature lints to do the self-build
export RUSTC_BOOTSTRAP_KEY:=$(CFG_BOOTSTRAP_KEY)
endif
export CFG_BOOTSTRAP_KEY

######################################################################
# Per-stage targets and runner
######################################################################

# Valid setting-strings are 'all', 'none', 'gdb', 'lldb'
# This 'function' will determine which debugger scripts to copy based on a
# target triple. See debuggers.mk for more information.
TRIPLE_TO_DEBUGGER_SCRIPT_SETTING=\
 $(if $(findstring windows,$(1)),none,$(if $(findstring darwin,$(1)),lldb,gdb))

STAGES = 0 1 2 3

define SREQ
# $(1) is the stage number
# $(2) is the target triple
# $(3) is the host triple

# Destinations of artifacts for the host compiler
HROOT$(1)_H_$(3) = $(3)/stage$(1)
HBIN$(1)_H_$(3) = $$(HROOT$(1)_H_$(3))/bin

ifeq ($$(CFG_WINDOWSY_$(3)),1)
# On Windows we always store host runtime libraries in the 'bin' directory because
# there's no rpath. Target libraries go under $CFG_LIBDIR_RELATIVE (usually 'lib').
HLIB_RELATIVE$(1)_H_$(3) = bin
TROOT$(1)_T_$(2)_H_$(3) = $$(HROOT$(1)_H_$(3))/$$(CFG_LIBDIR_RELATIVE)/rustlib/$(2)
# Remove the next 3 lines after a snapshot
ifeq ($(1),0)
RUSTFLAGS_STAGE0 += -L $$(TROOT$(1)_T_$(2)_H_$(3))/lib
endif

else

ifeq ($(1),0)
HLIB_RELATIVE$(1)_H_$(3) = lib
else
HLIB_RELATIVE$(1)_H_$(3) = $$(CFG_LIBDIR_RELATIVE)
endif
TROOT$(1)_T_$(2)_H_$(3) = $$(HLIB$(1)_H_$(3))/rustlib/$(2)

endif
HLIB$(1)_H_$(3) = $$(HROOT$(1)_H_$(3))/$$(HLIB_RELATIVE$(1)_H_$(3))

# Destinations of artifacts for target architectures
TBIN$(1)_T_$(2)_H_$(3) = $$(TROOT$(1)_T_$(2)_H_$(3))/bin
TLIB$(1)_T_$(2)_H_$(3) = $$(TROOT$(1)_T_$(2)_H_$(3))/lib

# Preqrequisites for using the stageN compiler
ifeq ($(1),0)
HSREQ$(1)_H_$(3) = $$(HBIN$(1)_H_$(3))/rustc$$(X_$(3))
else
HSREQ$(1)_H_$(3) = \
	$$(HBIN$(1)_H_$(3))/rustc$$(X_$(3)) \
	$$(MKFILE_DEPS) \
	tmp/install-debugger-scripts$(1)_H_$(3)-$$(call TRIPLE_TO_DEBUGGER_SCRIPT_SETTING,$(3)).done
endif

# Prerequisites for using the stageN compiler to build target artifacts
TSREQ$(1)_T_$(2)_H_$(3) = \
	$$(HSREQ$(1)_H_$(3)) \
	$$(foreach obj,$$(REQUIRED_OBJECTS_$(2)),\
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(obj))

# Prerequisites for a working stageN compiler and libraries, for a specific
# target
SREQ$(1)_T_$(2)_H_$(3) = \
	$$(TSREQ$(1)_T_$(2)_H_$(3)) \
	$$(foreach dep,$$(TARGET_CRATES), \
	    $$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(dep)) \
	tmp/install-debugger-scripts$(1)_T_$(2)_H_$(3)-$$(call TRIPLE_TO_DEBUGGER_SCRIPT_SETTING,$(2)).done

# Prerequisites for a working stageN compiler and complete set of target
# libraries
CSREQ$(1)_T_$(2)_H_$(3) = \
	$$(TSREQ$(1)_T_$(2)_H_$(3)) \
	$$(HBIN$(1)_H_$(3))/rustdoc$$(X_$(3)) \
	$$(foreach dep,$$(CRATES),$$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(dep))

ifeq ($(1),0)
# Don't run the stage0 compiler under valgrind - that ship has sailed
CFG_VALGRIND_COMPILE$(1) =
else
CFG_VALGRIND_COMPILE$(1) = $$(CFG_VALGRIND_COMPILE)
endif

# Add RUSTFLAGS_STAGEN values to the build command
EXTRAFLAGS_STAGE$(1) = $$(RUSTFLAGS_STAGE$(1))

CFGFLAG$(1)_T_$(2)_H_$(3) = stage$(1)

endef

# Same macro/variables as above, but defined in a separate loop so it can use
# all the variables above for all archs. The RPATH_VAR setup sometimes needs to
# reach across triples to get things in order.
#
# Defines (with the standard $(1)_T_$(2)_H_$(3) suffix):
# * `LD_LIBRARY_PATH_ENV_NAME`: the name for the key to use in the OS
#   environment to access or extend the lookup path for dynamic
#   libraries.  Note on Windows, that key is `$PATH`, and thus not
#   only conflates programs with dynamic libraries, but also often
#   contains spaces which confuse make.
# * `LD_LIBRARY_PATH_ENV_HOSTDIR`: the entry to add to lookup path for the host
# * `LD_LIBRARY_PATH_ENV_TARGETDIR`: the entry to add to lookup path for target
#
# Below that, HOST_RPATH_VAR and TARGET_RPATH_VAR are defined in terms of the
# above settings.
#
define SREQ_CMDS

ifeq ($$(OSTYPE_$(3)),apple-darwin)
  LD_LIBRARY_PATH_ENV_NAME$(1)_T_$(2)_H_$(3) := DYLD_LIBRARY_PATH
else
ifeq ($$(CFG_WINDOWSY_$(3)),1)
  LD_LIBRARY_PATH_ENV_NAME$(1)_T_$(2)_H_$(3) := PATH
else
  LD_LIBRARY_PATH_ENV_NAME$(1)_T_$(2)_H_$(3) := LD_LIBRARY_PATH
endif
endif

LD_LIBRARY_PATH_ENV_HOSTDIR$(1)_T_$(2)_H_$(3) := \
    $$(CURDIR)/$$(HLIB$(1)_H_$(3))
LD_LIBRARY_PATH_ENV_TARGETDIR$(1)_T_$(2)_H_$(3) := \
    $$(CURDIR)/$$(TLIB1_T_$(2)_H_$(CFG_BUILD))

HOST_RPATH_VAR$(1)_T_$(2)_H_$(3) := \
  $$(LD_LIBRARY_PATH_ENV_NAME$(1)_T_$(2)_H_$(3))=$$(LD_LIBRARY_PATH_ENV_HOSTDIR$(1)_T_$(2)_H_$(3)):$$$$$$(LD_LIBRARY_PATH_ENV_NAME$(1)_T_$(2)_H_$(3))
TARGET_RPATH_VAR$(1)_T_$(2)_H_$(3) := \
  $$(LD_LIBRARY_PATH_ENV_NAME$(1)_T_$(2)_H_$(3))=$$(LD_LIBRARY_PATH_ENV_TARGETDIR$(1)_T_$(2)_H_$(3)):$$$$$$(LD_LIBRARY_PATH_ENV_NAME$(1)_T_$(2)_H_$(3))

RPATH_VAR$(1)_T_$(2)_H_$(3) := $$(HOST_RPATH_VAR$(1)_T_$(2)_H_$(3))

# Pass --cfg stage0 only for the build->host part of stage0;
# if you're building a cross config, the host->* parts are
# effectively stage1, since it uses the just-built stage0.
#
# This logic is similar to how the LD_LIBRARY_PATH variable must
# change be slightly different when doing cross compilations.
# The build doesn't copy over all target libraries into
# a new directory, so we need to point the library path at
# the build directory where all the target libraries came
# from (the stage0 build host). Otherwise the relative rpaths
# inside of the rustc binary won't get resolved correctly.
ifeq ($(1),0)
ifneq ($(strip $(CFG_BUILD)),$(strip $(3)))
CFGFLAG$(1)_T_$(2)_H_$(3) = stage1

RPATH_VAR$(1)_T_$(2)_H_$(3) := $$(TARGET_RPATH_VAR$(1)_T_$(2)_H_$(3))
endif
endif

STAGE$(1)_T_$(2)_H_$(3) := \
	$$(Q)$$(RPATH_VAR$(1)_T_$(2)_H_$(3)) \
		$$(call CFG_RUN_TARG_$(3),$(1), \
		$$(CFG_VALGRIND_COMPILE$(1)) \
		$$(HBIN$(1)_H_$(3))/rustc$$(X_$(3)) \
		--cfg $$(CFGFLAG$(1)_T_$(2)_H_$(3)) \
		$$(CFG_RUSTC_FLAGS) $$(EXTRAFLAGS_STAGE$(1)) --target=$(2)) \
                $$(RUSTC_FLAGS_$(2))

PERF_STAGE$(1)_T_$(2)_H_$(3) := \
	$$(Q)$$(call CFG_RUN_TARG_$(3),$(1), \
		$$(CFG_PERF_TOOL) \
		$$(HBIN$(1)_H_$(3))/rustc$$(X_$(3)) \
		--cfg $$(CFGFLAG$(1)_T_$(2)_H_$(3)) \
		$$(CFG_RUSTC_FLAGS) $$(EXTRAFLAGS_STAGE$(1)) --target=$(2)) \
                $$(RUSTC_FLAGS_$(2))

endef

$(foreach build,$(CFG_HOST), \
 $(eval $(foreach target,$(CFG_TARGET), \
  $(eval $(foreach stage,$(STAGES), \
   $(eval $(call SREQ,$(stage),$(target),$(build))))))))

$(foreach build,$(CFG_HOST), \
 $(eval $(foreach target,$(CFG_TARGET), \
  $(eval $(foreach stage,$(STAGES), \
   $(eval $(call SREQ_CMDS,$(stage),$(target),$(build))))))))

######################################################################
# rustc-H-targets
#
# Builds a functional Rustc for the given host.
######################################################################

define DEF_RUSTC_STAGE_TARGET
# $(1) == architecture
# $(2) == stage

rustc-stage$(2)-H-$(1): \
	$$(foreach target,$$(CFG_TARGET),$$(SREQ$(2)_T_$$(target)_H_$(1)))

endef

$(foreach host,$(CFG_HOST), \
 $(eval $(foreach stage,1 2 3, \
  $(eval $(call DEF_RUSTC_STAGE_TARGET,$(host),$(stage))))))

rustc-stage1: rustc-stage1-H-$(CFG_BUILD)
rustc-stage2: rustc-stage2-H-$(CFG_BUILD)
rustc-stage3: rustc-stage3-H-$(CFG_BUILD)

define DEF_RUSTC_TARGET
# $(1) == architecture

rustc-H-$(1): rustc-stage2-H-$(1)
endef

$(foreach host,$(CFG_TARGET), \
 $(eval $(call DEF_RUSTC_TARGET,$(host))))

rustc-stage1: rustc-stage1-H-$(CFG_BUILD)
rustc-stage2: rustc-stage2-H-$(CFG_BUILD)
rustc-stage3: rustc-stage3-H-$(CFG_BUILD)
rustc: rustc-H-$(CFG_BUILD)

rustc-H-all: $(foreach host,$(CFG_HOST),rustc-H-$(host))

######################################################################
# Entrypoint rule
######################################################################

.DEFAULT_GOAL := all

define ALL_TARGET_N
ifneq ($$(findstring $(1),$$(CFG_HOST)),)
# This is a host
all-target-$(1)-host-$(2): $$(CSREQ2_T_$(1)_H_$(2))
else
# This is a target only
all-target-$(1)-host-$(2): $$(SREQ2_T_$(1)_H_$(2))
endif
endef

$(foreach target,$(CFG_TARGET), \
 $(foreach host,$(CFG_HOST), \
 $(eval $(call ALL_TARGET_N,$(target),$(host)))))

ALL_TARGET_RULES = $(foreach target,$(CFG_TARGET), \
	$(foreach host,$(CFG_HOST), \
 all-target-$(target)-host-$(host)))

all: $(ALL_TARGET_RULES) $(GENERATED) docs

######################################################################
# Build system documentation
######################################################################

# $(1) is the name of the doc <section> in Makefile.in
# pick everything between tags | remove first line | remove last line
# | remove extra (?) line | strip leading `#` from lines
SHOW_DOCS = $(Q)awk '/<$(1)>/,/<\/$(1)>/' $(S)/Makefile.in | sed '1d' | sed '$$d' | sed 's/^\# \?//'

help:
	$(call SHOW_DOCS,help)

tips:
	$(call SHOW_DOCS,tips)

nitty-gritty:
	$(call SHOW_DOCS,nitty-gritty)
