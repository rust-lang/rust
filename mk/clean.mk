# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

######################################################################
# Cleanup
######################################################################

CLEAN_STAGE_RULES := \
 $(foreach stage, $(STAGES), \
  $(foreach host, $(CFG_HOST), \
   clean$(stage)_H_$(host) \
   $(foreach target, $(CFG_TARGET), \
    clean$(stage)_T_$(target)_H_$(host))))

CLEAN_STAGE_RULES := $(CLEAN_STAGE_RULES) \
    $(foreach host, $(CFG_HOST), clean-generic-H-$(host))

CLEAN_STAGE_RULES := $(CLEAN_STAGE_RULES) \
    $(foreach host, $(CFG_TARGET), clean-generic-T-$(host))

CLEAN_LLVM_RULES = \
 $(foreach target, $(CFG_HOST), \
  clean-llvm$(target))

.PHONY: clean clean-all clean-misc clean-llvm

clean-all: clean clean-llvm

clean-llvm: $(CLEAN_LLVM_RULES)

clean: clean-misc $(CLEAN_STAGE_RULES)

clean-misc:
	@$(call E, cleaning)
	$(Q)rm -f $(RUNTIME_OBJS) $(RUNTIME_DEF)
	$(Q)rm -f $(RUSTLLVM_LIB_OBJS) $(RUSTLLVM_OBJS_OBJS) $(RUSTLLVM_DEF)
	$(Q)rm -Rf $(GENERATED)
	$(Q)rm -Rf tmp/*
	$(Q)rm -Rf rust-stage0-*.tar.bz2 $(PKG_NAME)-*.tar.gz $(PKG_NAME)-*.exe
	$(Q)rm -Rf dist/*
	$(Q)rm -Rf doc

define CLEAN_GENERIC

clean-generic-$(2)-$(1):
	$(Q)find $(1)/rustllvm \
	         $(1)/rt \
		 $(1)/test \
		 $(1)/stage* \
		 -type f \( \
         -name '*.[odasS]' -o \
         -name '*.so' -o \
         -name '*.dylib' -o \
         -name '*.rlib' -o \
         -name 'stamp.*' -o \
         -name '*.lib' -o \
         -name '*.dll' -o \
         -name '*.def' -o \
         -name '*.py' -o \
         -name '*.pyc' -o \
         -name '*.bc' -o \
         -name '*.rs' \
         \) \
         | xargs rm -f
	$(Q)find $(1) \
         -name '*.dSYM' \
         | xargs rm -Rf
endef

$(foreach host, $(CFG_HOST), $(eval $(call CLEAN_GENERIC,$(host),H)))
$(foreach targ, $(CFG_TARGET), $(eval $(call CLEAN_GENERIC,$(targ),T)))

define CLEAN_HOST_STAGE_N

clean$(1)_H_$(2): \
	    $$(foreach crate,$$(CRATES),clean$(1)_H_$(2)-lib-$$(crate)) \
	    $$(foreach tool,$$(TOOLS) $$(DEBUGGER_BIN_SCRIPTS_ALL),clean$(1)_H_$(2)-tool-$$(tool))
	$$(Q)rm -fr $(2)/rt/libbacktrace

clean$(1)_H_$(2)-tool-%:
	$$(Q)rm -f $$(HBIN$(1)_H_$(2))/$$*$$(X_$(2))

clean$(1)_H_$(2)-lib-%:
	$$(Q)rm -f $$(HLIB$(1)_H_$(2))/$$(call CFG_LIB_GLOB_$(2),$$*)
	$$(Q)rm -f $$(HLIB$(1)_H_$(2))/$$(call CFG_RLIB_GLOB,$$*)

endef

$(foreach host, $(CFG_HOST), \
 $(eval $(foreach stage, $(STAGES), \
  $(eval $(call CLEAN_HOST_STAGE_N,$(stage),$(host))))))

define CLEAN_TARGET_STAGE_N

clean$(1)_T_$(2)_H_$(3): \
	    $$(foreach crate,$$(CRATES),clean$(1)_T_$(2)_H_$(3)-lib-$$(crate)) \
	    $$(foreach tool,$$(TOOLS) $$(DEBUGGER_BIN_SCRIPTS_ALL),clean$(1)_T_$(2)_H_$(3)-tool-$$(tool))
	$$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/libcompiler-rt.a
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/librun_pass_stage* # For unix
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/run_pass_stage* # For windows

clean$(1)_T_$(2)_H_$(3)-tool-%:
	$$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/$$*$$(X_$(2))

clean$(1)_T_$(2)_H_$(3)-lib-%:
	$$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$$(call CFG_LIB_GLOB_$(2),$$*)
	$$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$$(call CFG_RLIB_GLOB,$$*)
endef

$(foreach host, $(CFG_HOST), \
 $(eval $(foreach target, $(CFG_TARGET), \
  $(eval $(foreach stage, 0 1 2 3, \
   $(eval $(call CLEAN_TARGET_STAGE_N,$(stage),$(target),$(host))))))))
