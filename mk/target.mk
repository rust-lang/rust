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
export CFG_COMPILER

# The standard libraries should be held up to a higher standard than any old
# code, make sure that these common warnings are denied by default. These can
# be overridden during development temporarily. For stage0, we allow warnings
# which may be bugs in stage0 (should be fixed in stage1+)
WFLAGS_ST0 = -W warnings
WFLAGS_ST1 = -D warnings
WFLAGS_ST2 = -D warnings

# TARGET_STAGE_N template: This defines how target artifacts are built
# for all stage/target architecture combinations. The arguments:
# $(1) is the stage
# $(2) is the target triple
# $(3) is the host triple

# Every recipe in TARGET_STAGE_N outputs to $$(TLIB$(1)_T_$(2)_H_$(3),
# a directory that can be cleaned out during the middle of a run of
# the get-snapshot.py script.  Therefore, every recipe needs to have
# an order-only dependency either on $(SNAPSHOT_RUSTC_POST_CLEANUP) or
# on $$(TSREQ$(1)_T_$(2)_H_$(3)), to ensure that no products will be
# put into the target area until after the get-snapshot.py script has
# had its chance to clean it out; otherwise the other products will be
# inadvertantly included in the clean out.

SNAPSHOT_RUSTC_POST_CLEANUP=$(HBIN0_H_$(CFG_BUILD))/rustc$(X_$(CFG_BUILD))

define TARGET_STAGE_N

$$(TLIB$(1)_T_$(2)_H_$(3))/libmorestack.a: \
		$(2)/rt/stage$(1)/arch/$$(HOST_$(2))/libmorestack.a \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/ \
		  $(SNAPSHOT_RUSTC_POST_CLEANUP)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_RUNTIME_$(2)): \
		$(2)/rt/stage$(1)/$(CFG_RUNTIME_$(2)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/ \
		  $(SNAPSHOT_RUSTC_POST_CLEANUP)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_STDLIB_$(2)): \
		$$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
		$$(TSREQ$(1)_T_$(2)_H_$(3)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
ifeq ($(1),0)
	$$(Q)cp $(3)/stage0/$(CFG_LIBDIR)/rustlib/$(2)/$(CFG_LIBDIR)/* \
		$(3)/stage0/$(CFG_LIBDIR)/rustc/$(2)/$(CFG_LIBDIR)/
endif
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_GLOB_$(2)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_RGLOB_$(2)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(2)_H_$(3)) $$(WFLAGS_ST$(1)) --out-dir $$(@D) $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_GLOB_$(2)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_RGLOB_$(2)),$$(notdir $$@))

$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_EXTRALIB_$(2)): \
		$$(EXTRALIB_CRATE) $$(EXTRALIB_INPUTS) \
	        $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_STDLIB_$(2)) \
		$$(TSREQ$(1)_T_$(2)_H_$(3)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
ifeq ($(1),0)
	$$(Q)cp $(3)/stage0/$(CFG_LIBDIR)/rustlib/$(2)/$(CFG_LIBDIR)/* \
		$(3)/stage0/$(CFG_LIBDIR)/rustc/$(2)/$(CFG_LIBDIR)/
endif
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_GLOB_$(2)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_RGLOB_$(2)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(2)_H_$(3)) $$(WFLAGS_ST$(1)) --out-dir $$(@D) $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_GLOB_$(2)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_RGLOB_$(2)),$$(notdir $$@))

$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTUV_$(2)): \
		$$(LIBRUSTUV_CRATE) $$(LIBRUSTUV_INPUTS) \
	        $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_STDLIB_$(2)) \
		$$(TSREQ$(1)_T_$(2)_H_$(3)) \
		$$(LIBUV_LIB_$(2)) \
		$$(UV_SUPPORT_LIB_$(2)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_GLOB_$(2)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_RGLOB_$(2)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(2)_H_$(3)) $$(WFLAGS_ST$(1)) \
		-L $$(UV_SUPPORT_DIR_$(2)) \
		-L $$(dir $$(LIBUV_LIB_$(2))) \
		--out-dir $$(@D) $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_GLOB_$(2)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_RGLOB_$(2)),$$(notdir $$@))

$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBGREEN_$(2)): \
		$$(LIBGREEN_CRATE) $$(LIBGREEN_INPUTS) \
	        $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_STDLIB_$(2)) \
		$$(TSREQ$(1)_T_$(2)_H_$(3)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBGREEN_GLOB_$(2)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBGREEN_RGLOB_$(2)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(2)_H_$(3)) $$(WFLAGS_ST$(1)) \
		--out-dir $$(@D) $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBGREEN_GLOB_$(2)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBGREEN_RGLOB_$(2)),$$(notdir $$@))

$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBNATIVE_$(2)): \
		$$(LIBNATIVE_CRATE) $$(LIBNATIVE_INPUTS) \
	        $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_STDLIB_$(2)) \
		$$(TSREQ$(1)_T_$(2)_H_$(3)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBNATIVE_GLOB_$(2)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBNATIVE_RGLOB_$(2)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(2)_H_$(3)) $$(WFLAGS_ST$(1)) \
		--out-dir $$(@D) $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBNATIVE_GLOB_$(2)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBNATIVE_RGLOB_$(2)),$$(notdir $$@))

$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBSYNTAX_$(3)): \
                $$(LIBSYNTAX_CRATE) $$(LIBSYNTAX_INPUTS) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))			\
		$$(TSTDLIB_DEFAULT$(1)_T_$(2)_H_$(3))      \
		$$(TEXTRALIB_DEFAULT$(1)_T_$(2)_H_$(3)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
ifeq ($(1),0)
	$$(Q)cp $(3)/stage0/$(CFG_LIBDIR)/rustlib/$(2)/$(CFG_LIBDIR)/* \
		$(3)/stage0/$(CFG_LIBDIR)/rustc/$(2)/$(CFG_LIBDIR)/
endif
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBSYNTAX_GLOB_$(2)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBSYNTAX_RGLOB_$(2)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(2)_H_$(3)) $$(WFLAGS_ST$(1)) $(BORROWCK) --out-dir $$(@D) $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBSYNTAX_GLOB_$(2)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBSYNTAX_RGLOB_$(2)),$$(notdir $$@))

# Only build the compiler for host triples
ifneq ($$(findstring $(2),$$(CFG_HOST)),)

$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_RUSTLLVM_$(3)): \
		$(2)/rustllvm/$(CFG_RUSTLLVM_$(3)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/ \
		  $(SNAPSHOT_RUSTC_POST_CLEANUP)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTC_$(3)): CFG_COMPILER = $(2)
$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTC_$(3)):		\
		$$(COMPILER_CRATE) $$(COMPILER_INPUTS)		\
		$(S)src/librustc/lib/llvmdeps.rs		\
		$$(TSREQ$(1)_T_$(2)_H_$(3)) \
                $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBSYNTAX_$(3)) \
                $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_RUSTLLVM_$(3)) \
		| $$(TLIB$(1)_T_$(2)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
ifeq ($(1),0)
	$$(Q)cp $(3)/stage0/$(CFG_LIBDIR)/rustlib/$(2)/$(CFG_LIBDIR)/* \
		$(3)/stage0/$(CFG_LIBDIR)/rustc/$(2)/$(CFG_LIBDIR)/
endif
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTC_GLOB_$(2)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTC_RGLOB_$(2)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(2)_H_$(3)) $$(WFLAGS_ST$(1)) \
	    -L "$$(LLVM_LIBDIR_$(2))" \
	    --out-dir $$(@D) $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTC_GLOB_$(2)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTC_RGLOB_$(2)),$$(notdir $$@))

$$(TBIN$(1)_T_$(2)_H_$(3))/rustc$$(X_$(3)):			\
		$$(DRIVER_CRATE)				\
		$$(SREQ$(1)_T_$(2)_H_$(3)) \
		$$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTC_$(3)) \
		| $$(TBIN$(1)_T_$(2)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
ifeq ($(1),0)
	$$(Q)cp $(3)/stage0/$(CFG_LIBDIR)/rustlib/$(2)/$(CFG_LIBDIR)/* \
		$(3)/stage0/$(CFG_LIBDIR)/rustc/$(2)/$(CFG_LIBDIR)/
endif
	$$(STAGE$(1)_T_$(2)_H_$(3)) --cfg rustc -o $$@ $$<
ifdef CFG_ENABLE_PAX_FLAGS
	@$$(call E, apply PaX flags: $$@)
	@"$(CFG_PAXCTL)" -cm "$$@"
endif

endif

$$(TBIN$(1)_T_$(2)_H_$(3))/:
	mkdir -p $$@

ifneq ($(CFG_LIBDIR),bin)
$$(TLIB$(1)_T_$(2)_H_$(3))/:
	mkdir -p $$@
endif

endef

# In principle, each host can build each target:
$(foreach source,$(CFG_HOST),				\
 $(foreach target,$(CFG_TARGET),			\
  $(eval $(call TARGET_STAGE_N,0,$(target),$(source)))		\
  $(eval $(call TARGET_STAGE_N,1,$(target),$(source)))		\
  $(eval $(call TARGET_STAGE_N,2,$(target),$(source)))		\
  $(eval $(call TARGET_STAGE_N,3,$(target),$(source)))))
