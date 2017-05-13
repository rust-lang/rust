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

# Used as defaults for the runtime ar and cc tools
export CFG_DEFAULT_LINKER
export CFG_DEFAULT_AR

# Macro that generates the full list of dependencies for a crate at a particular
# stage/target/host tuple.
#
# $(1) - stage
# $(2) - target
# $(3) - host
# $(4) crate
define RUST_CRATE_FULLDEPS
CRATE_FULLDEPS_$(1)_T_$(2)_H_$(3)_$(4) := \
		$$(CRATEFILE_$(4)) \
		$$(RSINPUTS_$(4)) \
		$$(foreach dep,$$(RUST_DEPS_$(4)_T_$(2)), \
		  $$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(dep)) \
		$$(foreach dep,$$(NATIVE_DEPS_$(4)), \
		  $$(RT_OUTPUT_DIR_$(2))/$$(call CFG_STATIC_LIB_NAME_$(2),$$(dep))) \
		$$(foreach dep,$$(NATIVE_DEPS_$(4)_T_$(2)), \
		  $$(RT_OUTPUT_DIR_$(2))/$$(dep))
endef

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
   $(foreach crate,$(CRATES), \
    $(eval $(call RUST_CRATE_FULLDEPS,$(stage),$(target),$(host),$(crate)))))))

# RUST_TARGET_STAGE_N template: This defines how target artifacts are built
# for all stage/target architecture combinations. This is one giant rule which
# works as follows:
#
#   1. The immediate dependencies are the rust source files
#   2. Each rust crate dependency is listed (based on their stamp files),
#      as well as all native dependencies (listed in RT_OUTPUT_DIR)
#   3. The stage (n-1) compiler is required through the TSREQ dependency
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
$$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$(4): \
		$$(CRATEFILE_$(4)) \
		$$(CRATE_FULLDEPS_$(1)_T_$(2)_H_$(3)_$(4)) \
		$$(LLVM_CONFIG_$(2)) \
		$$(TSREQ$(1)_T_$(2)_H_$(3)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, rustc: $$(@D)/lib$(4))
	@touch $$@.start_time
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES, \
	    $$(dir $$@)$$(call CFG_LIB_GLOB_$(2),$(4)))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES, \
	    $$(dir $$@)$$(call CFG_RLIB_GLOB,$(4)))
	$(Q)CFG_LLVM_LINKAGE_FILE=$$(LLVM_LINKAGE_PATH_$(2)) \
	    $$(subst @,,$$(STAGE$(1)_T_$(2)_H_$(3))) \
		$$(RUST_LIB_FLAGS_ST$(1)) \
		-L "$$(RT_OUTPUT_DIR_$(2))" \
		$$(LLVM_LIBDIR_RUSTFLAGS_$(2)) \
		$$(LLVM_STDCPP_RUSTFLAGS_$(2)) \
		$$(RUSTFLAGS_$(4)) \
		$$(RUSTFLAGS$(1)_$(4)) \
		$$(RUSTFLAGS$(1)_$(4)_T_$(2)) \
		--out-dir $$(@D) \
		-C extra-filename=-$$(CFG_FILENAME_EXTRA) \
		-C metadata=$$(CFG_FILENAME_EXTRA) \
		$$<
	@touch -r $$@.start_time $$@ && rm $$@.start_time
	$$(call LIST_ALL_OLD_GLOB_MATCHES, \
	    $$(dir $$@)$$(call CFG_LIB_GLOB_$(2),$(4)))
	$$(call LIST_ALL_OLD_GLOB_MATCHES, \
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

$$(TBIN$(1)_T_$(2)_H_$(3))/$(4)$$(X_$(2)): \
		$$(TOOL_SOURCE_$(4)) \
		$$(TOOL_INPUTS_$(4)) \
		$$(foreach dep,$$(TOOL_DEPS_$(4)), \
		    $$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(dep)) \
		$$(TSREQ$(1)_T_$(2)_H_$(3)) \
		| $$(TBIN$(1)_T_$(2)_H_$(3))/
	@$$(call E, rustc: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) \
		$$(LLVM_LIBDIR_RUSTFLAGS_$(2)) \
		-o $$@ $$< --cfg $(4)

endef

# Macro for building runtime startup/shutdown object files;
# these are Rust's equivalent of crti.o, crtn.o
#
# $(1) - stage
# $(2) - target triple
# $(3) - host triple
# $(4) - object basename
define TARGET_RUSTRT_STARTUP_OBJ

$$(TLIB$(1)_T_$(2)_H_$(3))/$(4).o: \
		$(S)src/rtstartup/$(4).rs \
		$$(TLIB$(1)_T_$(2)_H_$(3))/stamp.core \
		$$(HSREQ$(1)_T_$(2)_H_$(3)) \
		| $$(TBIN$(1)_T_$(2)_H_$(3))/
	@$$(call E, rustc: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) --emit=obj -o $$@ $$<

ifeq ($$(CFG_RUSTRT_HAS_STARTUP_OBJS_$(2)), 1)
# Add dependencies on Rust startup objects to all crates that depend on core.
# This ensures that they are built after core (since they depend on it),
# but before everything else (since they are needed for linking dylib crates).
$$(foreach crate, $$(TARGET_CRATES_$(2)), \
	$$(if $$(findstring core,$$(DEPS_$$(crate))), \
		$$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(crate))) : $$(TLIB$(1)_T_$(2)_H_$(3))/$(4).o
endif

endef

# Every recipe in RUST_TARGET_STAGE_N outputs to $$(TLIB$(1)_T_$(2)_H_$(3),
# a directory that can be cleaned out during the middle of a run of
# the get-snapshot.py script.  Therefore, every recipe needs to have
# an order-only dependency either on $(SNAPSHOT_RUSTC_POST_CLEANUP) or
# on $$(TSREQ$(1)_T_$(2)_H_$(3)), to ensure that no products will be
# put into the target area until after the get-snapshot.py script has
# had its chance to clean it out; otherwise the other products will be
# inadvertently included in the clean out.
SNAPSHOT_RUSTC_POST_CLEANUP=$(HBIN0_H_$(CFG_BUILD))/rustc$(X_$(CFG_BUILD))

define TARGET_HOST_RULES

$$(TLIB$(1)_T_$(2)_H_$(3))/: $$(SNAPSHOT_RUSTC_POST_CLEANUP)
	mkdir -p $$@

$$(TBIN$(1)_T_$(2)_H_$(3))/: $$(SNAPSHOT_RUSTC_POST_CLEANUP)
	mkdir -p $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/%: $$(RT_OUTPUT_DIR_$(2))/% \
	    $$(SNAPSHOT_RUSTC_POST_CLEANUP) \
	    | $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
endef

$(foreach source,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(eval $(call TARGET_HOST_RULES,0,$(target),$(source))) \
  $(eval $(call TARGET_HOST_RULES,1,$(target),$(source))) \
  $(eval $(call TARGET_HOST_RULES,2,$(target),$(source))) \
  $(eval $(call TARGET_HOST_RULES,3,$(target),$(source)))))

# In principle, each host can build each target for both libs and tools
$(foreach crate,$(CRATES), \
 $(foreach source,$(CFG_HOST), \
  $(foreach target,$(CFG_TARGET), \
   $(eval $(call RUST_TARGET_STAGE_N,0,$(target),$(source),$(crate))) \
   $(eval $(call RUST_TARGET_STAGE_N,1,$(target),$(source),$(crate))) \
   $(eval $(call RUST_TARGET_STAGE_N,2,$(target),$(source),$(crate))) \
   $(eval $(call RUST_TARGET_STAGE_N,3,$(target),$(source),$(crate))))))

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
   $(foreach tool,$(TOOLS), \
    $(eval $(call TARGET_TOOL,$(stage),$(target),$(host),$(tool)))))))

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
   $(foreach obj,rsbegin rsend, \
    $(eval $(call TARGET_RUSTRT_STARTUP_OBJ,$(stage),$(target),$(host),$(obj)))))))
