# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# TARGET_STAGE_N template: This defines how target artifacts are built
# for all stage/target architecture combinations. The arguments:
# $(1) is the stage
# $(2) is the target triple
# $(3) is the host triple


define TARGET_STAGE_N

$$(TLIB$(1)_T_$(2)_H_$(3))/libmorestack.a: \
		rt/$(2)/arch/$$(HOST_$(2))/libmorestack.a
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUSTLLVM): \
		rustllvm/$(2)/$$(CFG_RUSTLLVM)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBRUSTC):		\
		$$(COMPILER_CRATE) $$(COMPILER_INPUTS)		\
                $$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBSYNTAX)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< && touch $$@

$$(TBIN$(1)_T_$(2)_H_$(3))/rustc$$(X):			\
		$$(DRIVER_CRATE) 							\
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) --cfg rustc -o $$@ $$<
ifdef CFG_ENABLE_PAX_FLAGS
	@$$(call E, apply PaX flags: $$@)
	@"$(CFG_PAXCTL)" -cm "$$@"
endif

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBSYNTAX): \
                $$(LIBSYNTAX_CRATE) $$(LIBSYNTAX_INPUTS) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))			\
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUSTLLVM)	\
		$$(TCORELIB_DEFAULT$(1)_T_$(2)_H_$(3))      \
		$$(TSTDLIB_DEFAULT$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) $(BORROWCK) -o $$@ $$< && touch $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUNTIME): \
		rt/$(2)/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_CORELIB): \
		$$(CORELIB_CRATE) $$(CORELIB_INPUTS) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< && touch $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_STDLIB): \
		$$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
	        $$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_CORELIB) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< && touch $$@

endef

# In principle, each host can build each target:
$(foreach source,$(CFG_TARGET_TRIPLES),				\
 $(foreach target,$(CFG_TARGET_TRIPLES),			\
  $(eval $(call TARGET_STAGE_N,0,$(target),$(source)))		\
  $(eval $(call TARGET_STAGE_N,1,$(target),$(source)))		\
  $(eval $(call TARGET_STAGE_N,2,$(target),$(source)))		\
  $(eval $(call TARGET_STAGE_N,3,$(target),$(source)))))
