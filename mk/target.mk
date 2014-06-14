# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# This is the compile-time target-triple for the compiler. For the compiler at
# runtime, this should be considered the host-triple. More explanation for why
# this exists can be found on issue #2400
export CFG_COMPILER_HOST_TRIPLE

# The standard libraries should be held up to a higher standard than any old
# code, make sure that these common warnings are denied by default. These can
# be overridden during development temporarily. For stage0, we allow warnings
# which may be bugs in stage0 (should be fixed in stage1+)
WFLAGS_ST0 = -W warnings
WFLAGS_ST1 = -D warnings
WFLAGS_ST2 = -D warnings

# Macro that generates the full list of dependencies for a crate at a particular
# stage/target/host tuple.
#
# $(1) - stage
# $(2) - target
# $(3) - host
# $(4) crate
define RUST_CRATE_FULLDEPS
CRATE_FULLDEPS_$(1)_T_$(2)_H_$(3)_$(4) :=			    \
		$$(CRATEFILE_$(4))				    \
		$$(RSINPUTS_$(4))				    \
		$$(foreach dep,$$(RUST_DEPS_$(4)),		    \
		  $$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(dep))	    \
		$$(foreach dep,$$(NATIVE_DEPS_$(4)),		    \
		  $$(RT_OUTPUT_DIR_$(2))/$$(call CFG_STATIC_LIB_NAME_$(2),$$(dep)))
endef

$(foreach host,$(CFG_HOST),						    \
 $(foreach target,$(CFG_TARGET),					    \
  $(foreach stage,$(STAGES),						    \
   $(foreach crate,$(CRATES),						    \
    $(eval $(call RUST_CRATE_FULLDEPS,$(stage),$(target),$(host),$(crate)))))))

# RUST_TARGET_STAGE_N template: This defines how target artifacts are built
# for all stage/target architecture combinations. This is one giant rule which
# works as follows:
#
#   1. The immediate dependencies are the rust source files
#   2. Each rust crate dependency is listed (based on their stamp files),
#      as well as all native dependencies (listed in RT_OUTPUT_DIR)
#   3. The stage (n-1) compiler is required through the TSREQ dependency, along
#      with the morestack library
#   4. When actually executing the rule, the first thing we do is to clean out
#      old libs and rlibs via the REMOVE_ALL_OLD_GLOB_MATCHES macro
#   5. Finally, we get around to building the actual crate. It's just one
#      "small" invocation of the previous stage rustc. We use -L to
#      RT_OUTPUT_DIR so all the native dependencies are picked up.
#      Additionally, we pass in the llvm dir so rustc can link against it.
#   6. Some cleanup is done (listing what was just built) if verbose is turned
#      on.
#
# $(1) is the stage
# $(2) is the target triple
# $(3) is the host triple
# $(4) is the crate name
define RUST_TARGET_STAGE_N

$$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$(4): CFG_COMPILER_HOST_TRIPLE = $(2)
$$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$(4):				    \
		$$(CRATEFILE_$(4))				    \
		$$(CRATE_FULLDEPS_$(1)_T_$(2)_H_$(3)_$(4))	    \
		$$(TSREQ$(1)_T_$(2)_H_$(3))			    \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, rustc: $$(@D)/lib$(4))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES,\
	    $$(dir $$@)$$(call CFG_LIB_GLOB_$(2),$(4)))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES,\
	    $$(dir $$@)$$(call CFG_RLIB_GLOB,$(4)))
	$$(STAGE$(1)_T_$(2)_H_$(3)) \
		$$(WFLAGS_ST$(1)) \
		-L "$$(RT_OUTPUT_DIR_$(2))" \
		-L "$$(LLVM_LIBDIR_$(2))" \
		-L "$$(dir $$(LLVM_STDCPP_LOCATION_$(2)))" \
		$$(RUSTFLAGS_$(4)) \
		--out-dir $$(@D) $$<
	@touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES,\
	    $$(dir $$@)$$(call CFG_LIB_GLOB_$(2),$(4)))
	$$(call LIST_ALL_OLD_GLOB_MATCHES,\
	    $$(dir $$@)$$(call CFG_RLIB_GLOB,$(4)))

endef

# Macro for building any tool as part of the rust compilation process. Each
# tool is defined in crates.mk with a list of library dependencies as well as
# the source file for the tool. Building each tool will also be passed '--cfg
# <tool>' for usage in driver.rs
#
# This build rule is similar to the one found above, just tweaked for
# locations and things.
#
# $(1) - stage
# $(2) - target triple
# $(3) - host triple
# $(4) - name of the tool being built
define TARGET_TOOL

$$(TBIN$(1)_T_$(2)_H_$(3))/$(4)$$(X_$(2)):			\
		$$(TOOL_SOURCE_$(4))				\
		$$(TOOL_INPUTS_$(4))				\
		$$(foreach dep,$$(TOOL_DEPS_$(4)),		\
		    $$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(dep))	\
		$$(TSREQ$(1)_T_$(2)_H_$(3))			\
		| $$(TBIN$(1)_T_$(4)_H_$(3))/
	@$$(call E, rustc: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< --cfg $(4)

endef

# Every recipe in RUST_TARGET_STAGE_N outputs to $$(TLIB$(1)_T_$(2)_H_$(3),
# a directory that can be cleaned out during the middle of a run of
# the get-snapshot.py script.  Therefore, every recipe needs to have
# an order-only dependency either on $(SNAPSHOT_RUSTC_POST_CLEANUP) or
# on $$(TSREQ$(1)_T_$(2)_H_$(3)), to ensure that no products will be
# put into the target area until after the get-snapshot.py script has
# had its chance to clean it out; otherwise the other products will be
# inadvertantly included in the clean out.
SNAPSHOT_RUSTC_POST_CLEANUP=$(HBIN0_H_$(CFG_BUILD))/rustc$(X_$(CFG_BUILD))

define TARGET_HOST_RULES

$$(TLIB$(1)_T_$(2)_H_$(3))/stamp.rustc: $(S)src/librustc/lib/llvmdeps.rs

$$(TBIN$(1)_T_$(2)_H_$(3))/:
	mkdir -p $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/:
	mkdir -p $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/libcompiler-rt.a: \
	    $$(RT_OUTPUT_DIR_$(2))/$$(call CFG_STATIC_LIB_NAME_$(2),compiler-rt) \
	    | $$(TLIB$(1)_T_$(2)_H_$(3))/ $$(SNAPSHOT_RUSTC_POST_CLEANUP)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/libmorestack.a: \
	    $$(RT_OUTPUT_DIR_$(2))/$$(call CFG_STATIC_LIB_NAME_$(2),morestack) \
	    | $$(TLIB$(1)_T_$(2)_H_$(3))/ $$(SNAPSHOT_RUSTC_POST_CLEANUP)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
endef

$(foreach source,$(CFG_HOST),						    \
 $(foreach target,$(CFG_TARGET),					    \
  $(eval $(call TARGET_HOST_RULES,0,$(target),$(source)))		    \
  $(eval $(call TARGET_HOST_RULES,1,$(target),$(source)))		    \
  $(eval $(call TARGET_HOST_RULES,2,$(target),$(source)))		    \
  $(eval $(call TARGET_HOST_RULES,3,$(target),$(source)))))

# In principle, each host can build each target for both libs and tools
$(foreach crate,$(CRATES),						    \
 $(foreach source,$(CFG_HOST),						    \
  $(foreach target,$(CFG_TARGET),					    \
   $(eval $(call RUST_TARGET_STAGE_N,0,$(target),$(source),$(crate)))	    \
   $(eval $(call RUST_TARGET_STAGE_N,1,$(target),$(source),$(crate)))	    \
   $(eval $(call RUST_TARGET_STAGE_N,2,$(target),$(source),$(crate)))	    \
   $(eval $(call RUST_TARGET_STAGE_N,3,$(target),$(source),$(crate))))))

$(foreach host,$(CFG_HOST),						    \
 $(foreach target,$(CFG_TARGET),					    \
  $(foreach stage,$(STAGES),						    \
   $(foreach tool,$(TOOLS),						    \
    $(eval $(call TARGET_TOOL,$(stage),$(target),$(host),$(tool)))))))
