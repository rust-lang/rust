# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# CP_HOST_STAGE_N template: arg 1 is the N we're promoting *from*, arg
# 2 is N+1. Must be invoked to promote target artifacts to host
# artifacts for stage 1-3 (stage0 host artifacts come from the
# snapshot).  Arg 3 is the triple we're copying FROM and arg 4 is the
# triple we're copying TO.
#
# The easiest way to read this template is to assume we're promoting
# stage1 to stage2 and mentally gloss $(1) as 1, $(2) as 2.

define CP_HOST_STAGE_N

# Host libraries and executables (stage$(2)/bin/rustc and its runtime needs)

$$(HBIN$(2)_H_$(4))/rustc$$(X): \
	$$(TBIN$(1)_T_$(4)_H_$(3))/rustc$$(X) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUNTIME) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUSTLLVM) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTC) \
	$$(HCORELIB_DEFAULT$(2)_H_$(4)) \
	$$(HSTDLIB_DEFAULT$(2)_H_$(4))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$$(CFG_LIBRUSTC): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBRUSTC) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_LIBSYNTAX) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUNTIME) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUSTLLVM) \
	$$(HCORELIB_DEFAULT$(2)_H_$(3)) \
	$$(HSTDLIB_DEFAULT$(2)_H_$(3))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTC_GLOB) \
		$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTC_DSYM_GLOB)) \
	        $$(HLIB$(2)_H_$(4))

$$(HLIB$(2)_H_$(4))/$$(CFG_LIBSYNTAX): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_LIBSYNTAX) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUNTIME) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUSTLLVM) \
	$$(HCORELIB_DEFAULT$(2)_H_$(3)) \
	$$(HSTDLIB_DEFAULT$(2)_H_$(3))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBSYNTAX_GLOB) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBSYNTAX_DSYM_GLOB)) \
	        $$(HLIB$(2)_H_$(4))

$$(HLIB$(2)_H_$(4))/$$(CFG_RUNTIME): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$$(CFG_CORELIB): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_CORELIB) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
# Subtle: We do not let the shell expand $(CORELIB_DSYM_GLOB) directly rather
# we use Make's $$(wildcard) facility. The reason is that, on mac, when using
# USE_SNAPSHOT_CORELIB, we copy the core.dylib file out of the snapshot.
# In that case, there is no .dSYM file.  Annoyingly, bash then refuses to expand
# glob, and cp reports an error because libcore-*.dylib.dsym does not exist.
# Make instead expands the glob to nothing, which gives us the correct behavior.
# (Copy .dsym file if it exists, but do nothing otherwise)
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(CORELIB_GLOB) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(CORELIB_DSYM_GLOB)) \
	        $$(HLIB$(2)_H_$(4))

$$(HLIB$(2)_H_$(4))/$$(CFG_STDLIB): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_STDLIB) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_CORELIB) \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(STDLIB_GLOB) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(STDLIB_DSYM_GLOB)) \
	        $$(HLIB$(2)_H_$(4))

$$(HLIB$(2)_H_$(4))/libcore.rlib: \
	$$(TLIB$(1)_T_$(4)_H_$(3))/libcore.rlib \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/libstd.rlib: \
	$$(TLIB$(1)_T_$(4)_H_$(3))/libstd.rlib \
	$$(HLIB$(2)_H_$(4))/libcore.rlib \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/librustc.rlib: \
	$$(TLIB$(1)_T_$(4)_H_$(3))/librustc.rlib \
	$$(HLIB$(2)_H_$(4))/libcore.rlib \
	$$(HLIB$(2)_H_$(4))/libstd.rlib \
	$$(HLIB$(2)_H_$(4))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$$(CFG_RUSTLLVM): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$$(CFG_RUSTLLVM)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

$(foreach t,$(CFG_TARGET_TRIPLES),					\
	$(eval $(call CP_HOST_STAGE_N,0,1,$(t),$(t)))	\
	$(eval $(call CP_HOST_STAGE_N,1,2,$(t),$(t)))	\
	$(eval $(call CP_HOST_STAGE_N,2,3,$(t),$(t))))
