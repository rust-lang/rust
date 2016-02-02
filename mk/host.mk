# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Generic rule for copying any target crate to a host crate. This rule will also
# promote any dependent rust crates up to their host locations as well
#
# $(1) - the stage to copy from
# $(2) - the stage to copy to
# $(3) - the host triple
# $(4) - the target triple (same as $(3))
# $(5) - the name of the crate being processed
define CP_HOST_STAGE_N_CRATE

ifeq ($$(ONLY_RLIB_$(5)),)
$$(HLIB$(2)_H_$(4))/stamp.$(5): \
	$$(TLIB$(1)_T_$(3)_H_$(4))/stamp.$(5) \
	$$(RUST_DEPS_$(5):%=$$(HLIB$(2)_H_$(4))/stamp.%) \
	| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$(@D)/lib$(5))
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES, \
	    $$(dir $$@)$$(call CFG_LIB_GLOB_$(3),$(5)))
	$$(Q)cp $$< $$@
	$$(Q)cp -R $$(TLIB$(1)_T_$(3)_H_$(4))/$$(call CFG_LIB_GLOB_$(3),$(5)) \
	        $$(HLIB$(2)_H_$(4))
	$$(call LIST_ALL_OLD_GLOB_MATCHES, \
	    $$(dir $$@)$$(call CFG_LIB_GLOB_$(3),$(5)))
else
$$(HLIB$(2)_H_$(4))/stamp.$(5):
	$$(Q)touch $$@
endif

endef

# Same as the above macro, but for tools instead of crates
define CP_HOST_STAGE_N_TOOL

$$(HBIN$(2)_H_$(4))/$(5)$$(X_$(3)): \
	$$(TBIN$(1)_T_$(3)_H_$(4))/$(5)$$(X_$(3)) \
	$$(TOOL_DEPS_$(5):%=$$(HLIB$(2)_H_$(4))/stamp.%) \
	| $$(HBIN$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef


# Miscellaneous rules for just making a few directories.
#
# $(1) - the stage to copy from
# $(2) - the stage to copy to
# $(3) - the target triple
# $(4) - the host triple (same as $(3))
define CP_HOST_STAGE_N

ifneq ($(CFG_LIBDIR_RELATIVE),bin)
$$(HLIB$(2)_H_$(4))/:
	@mkdir -p $$@
endif

endef

$(foreach t,$(CFG_HOST), \
	$(eval $(call CP_HOST_STAGE_N,0,1,$(t),$(t))) \
	$(eval $(call CP_HOST_STAGE_N,1,2,$(t),$(t))) \
	$(eval $(call CP_HOST_STAGE_N,2,3,$(t),$(t))))

$(foreach crate,$(CRATES), \
 $(foreach t,$(CFG_HOST), \
  $(eval $(call CP_HOST_STAGE_N_CRATE,0,1,$(t),$(t),$(crate))) \
  $(eval $(call CP_HOST_STAGE_N_CRATE,1,2,$(t),$(t),$(crate))) \
  $(eval $(call CP_HOST_STAGE_N_CRATE,2,3,$(t),$(t),$(crate)))))

$(foreach tool,$(TOOLS), \
 $(foreach t,$(CFG_HOST), \
  $(eval $(call CP_HOST_STAGE_N_TOOL,0,1,$(t),$(t),$(tool))) \
  $(eval $(call CP_HOST_STAGE_N_TOOL,1,2,$(t),$(t),$(tool))) \
  $(eval $(call CP_HOST_STAGE_N_TOOL,2,3,$(t),$(t),$(tool)))))
