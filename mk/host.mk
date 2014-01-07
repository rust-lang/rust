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

# Note: $(3) and $(4) are both the same!

$$(HBIN$(2)_H_$(4))/rustc$$(X_$(4)): \
	$$(TBIN$(1)_T_$(4)_H_$(3))/rustc$$(X_$(4)) \
	$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTC_$(4)) \
	| $$(HBIN$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTC_$(4)): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTC_$(4)) \
	$$(HLIB$(2)_H_$(4))/$(CFG_LIBSYNTAX_$(4)) \
	| $$(HLIB$(2)_H_$(4))/

	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTC_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTC_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTC_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTC_GLOB_$(4)),$$(notdir $$@))

$$(HLIB$(2)_H_$(4))/$(CFG_LIBSYNTAX_$(4)): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBSYNTAX_$(4)) \
	$$(HSTDLIB_DEFAULT$(2)_H_$(4)) \
	$$(HEXTRALIB_DEFAULT$(2)_H_$(4)) \
	$$(HLIBRUSTUV_DEFAULT$(2)_H_$(4)) \
	$$(HLIBGREEN_DEFAULT$(2)_H_$(4)) \
	$$(HLIBNATIVE_DEFAULT$(2)_H_$(4)) \
	| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBSYNTAX_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBSYNTAX_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBSYNTAX_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBSYNTAX_GLOB_$(4)),$$(notdir $$@))

$$(HLIB$(2)_H_$(4))/$(CFG_RUNTIME_$(4)): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_RUNTIME_$(4)) \
	| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$(CFG_STDLIB_$(4)): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_STDLIB_$(4)) \
	| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_GLOB_$(4)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_RGLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
# Subtle: We do not let the shell expand $$(STDLIB_DSYM_GLOB) directly rather
# we use Make's $$(wildcard) facility. The reason is that, on mac, when using
# USE_SNAPSHOT_STDLIB, we copy the std.dylib file out of the snapshot.
# In that case, there is no .dSYM file.  Annoyingly, bash then refuses to expand
# glob, and cp reports an error because libstd-*.dylib.dsym does not exist.
# Make instead expands the glob to nothing, which gives us the correct behavior.
# (Copy .dsym file if it exists, but do nothing otherwise)
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(STDLIB_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(STDLIB_RGLOB_$(4))) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(STDLIB_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_GLOB_$(4)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_RGLOB_$(4)),$$(notdir $$@))

$$(HLIB$(2)_H_$(4))/$(CFG_EXTRALIB_$(4)): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_EXTRALIB_$(4)) \
	$$(HSTDLIB_DEFAULT$(2)_H_$(4)) \
	| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_GLOB_$(4)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_RGLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(EXTRALIB_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(EXTRALIB_RGLOB_$(4))) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(EXTRALIB_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_GLOB_$(4)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_RGLOB_$(4)),$$(notdir $$@))

$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTUV_$(4)): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTUV_$(4)) \
	$$(HLIB$(2)_H_$(4))/$(CFG_STDLIB_$(4)) \
	| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_GLOB_$(4)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_RGLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTUV_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTUV_RGLOB_$(4))) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTUV_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_GLOB_$(4)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_RGLOB_$(4)),$$(notdir $$@))

$$(HLIB$(2)_H_$(4))/$(CFG_LIBGREEN_$(4)): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBGREEN_$(4)) \
	$$(HLIB$(2)_H_$(4))/$(CFG_STDLIB_$(4)) \
	| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBGREEN_GLOB_$(4)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBGREEN_RGLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBGREEN_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBGREEN_RGLOB_$(4))) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBGREEN_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBGREEN_GLOB_$(4)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBGREEN_RGLOB_$(4)),$$(notdir $$@))

$$(HLIB$(2)_H_$(4))/$(CFG_LIBNATIVE_$(4)): \
	$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBNATIVE_$(4)) \
	$$(HLIB$(2)_H_$(4))/$(CFG_STDLIB_$(4)) \
	| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBNATIVE_GLOB_$(4)),$$(notdir $$@))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBNATIVE_RGLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBNATIVE_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBNATIVE_RGLOB_$(4))) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBNATIVE_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBNATIVE_GLOB_$(4)),$$(notdir $$@))
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBNATIVE_RGLOB_$(4)),$$(notdir $$@))

$$(HBIN$(2)_H_$(4))/:
	mkdir -p $$@

ifneq ($(LIBDIR_RELATIVE),bin)
$$(HLIB$(2)_H_$(4))/:
	mkdir -p $$@
endif

endef

$(foreach t,$(CFG_HOST),					\
	$(eval $(call CP_HOST_STAGE_N,0,1,$(t),$(t)))	\
	$(eval $(call CP_HOST_STAGE_N,1,2,$(t),$(t)))	\
	$(eval $(call CP_HOST_STAGE_N,2,3,$(t),$(t))))
