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

CLEAN_STAGE_RULES :=							\
 $(foreach stage, $(STAGES),						\
  $(foreach host, $(CFG_HOST),						\
   clean$(stage)_H_$(host)						\
   $(foreach target, $(CFG_TARGET),					\
    clean$(stage)_T_$(target)_H_$(host))))

CLEAN_STAGE_RULES := $(CLEAN_STAGE_RULES)				\
    $(foreach host, $(CFG_HOST), clean-generic-H-$(host))

CLEAN_STAGE_RULES := $(CLEAN_STAGE_RULES)				\
    $(foreach host, $(CFG_TARGET), clean-generic-T-$(host))

CLEAN_LLVM_RULES = 								\
 $(foreach target, $(CFG_HOST),		\
  clean-llvm$(target))

.PHONY: clean clean-all clean-misc clean-llvm

clean-all: clean clean-llvm

clean-llvm: $(CLEAN_LLVM_RULES)

clean: clean-misc $(CLEAN_STAGE_RULES)

clean-misc:
	@$(call E, cleaning)
	$(Q)rm -f $(RUNTIME_OBJS) $(RUNTIME_DEF)
	$(Q)rm -f $(RUSTLLVM_LIB_OBJS) $(RUSTLLVM_OBJS_OBJS) $(RUSTLLVM_DEF)
	$(Q)rm -Rf $(DOCS)
	$(Q)rm -Rf $(GENERATED)
	$(Q)rm -Rf tmp/*
	$(Q)rm -Rf rust-stage0-*.tar.bz2 $(PKG_NAME)-*.tar.gz dist
	$(Q)rm -Rf $(foreach ext, \
                 html aux cp fn ky log pdf pg toc tp vr cps epub, \
                 $(wildcard doc/*.$(ext)))
	$(Q)find doc/std doc/extra -mindepth 1 | xargs rm -Rf
	$(Q)rm -Rf doc/version.md
	$(Q)rm -Rf $(foreach sub, index styles files search javascript, \
                 $(wildcard doc/*/$(sub)))

define CLEAN_GENERIC

clean-generic-$(2)-$(1):
	$(Q)find $(1)/rustllvm \
	         $(1)/rt \
		 $(1)/test \
		 $(1)/stage* \
         -name '*.[odasS]' -o \
         -name '*.so' -o      \
         -name '*.dylib' -o   \
         -name '*.lib' -o     \
         -name '*.dll' -o     \
         -name '*.def' -o     \
         -name '*.bc'         \
         | xargs rm -f
	$(Q)find $(1)\
         -name '*.dSYM'       \
         | xargs rm -Rf
endef

$(foreach host, $(CFG_HOST), $(eval $(call CLEAN_GENERIC,$(host),H)))
$(foreach targ, $(CFG_TARGET), $(eval $(call CLEAN_GENERIC,$(targ),T)))

define CLEAN_HOST_STAGE_N

clean$(1)_H_$(2):
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rustc$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rustpkg$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/serializer$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rustdoc$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rust$(X_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTPKG_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTDOC_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_RUNTIME_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_STDLIB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_EXTRALIB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTUV_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTC_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBSYNTAX_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(STDLIB_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(STDLIB_RGLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(EXTRALIB_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(EXTRALIB_RGLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTUV_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTUV_RGLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTC_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBSYNTAX_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTPKG_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTDOC_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_RUSTLLVM_$(2))

endef

$(foreach host, $(CFG_HOST), \
 $(eval $(foreach stage, $(STAGES), \
  $(eval $(call CLEAN_HOST_STAGE_N,$(stage),$(host))))))

define CLEAN_TARGET_STAGE_N

clean$(1)_T_$(2)_H_$(3):
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rustc$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rustpkg$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/serializer$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rustdoc$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rust$(X_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTPKG_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTDOC_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_RUNTIME_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_STDLIB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_EXTRALIB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTUV_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTC_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBSYNTAX_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(STDLIB_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(STDLIB_RGLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(EXTRALIB_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(EXTRALIB_RGLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTUV_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTUV_RGLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTC_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTC_RGLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBSYNTAX_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBSYNTAX_RGLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTPKG_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTDOC_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_RUSTLLVM_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/libmorestack.a
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/librun_pass_stage* # For unix
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/run_pass_stage* # For windows
endef

$(foreach host, $(CFG_HOST), \
 $(eval $(foreach target, $(CFG_TARGET), \
  $(eval $(foreach stage, 0 1 2 3, \
   $(eval $(call CLEAN_TARGET_STAGE_N,$(stage),$(target),$(host))))))))

define DEF_CLEAN_LLVM_HOST
ifeq ($(CFG_LLVM_ROOT),)
clean-llvm$(1):
	$$(Q)$$(MAKE) -C $$(CFG_LLVM_BUILD_DIR_$(1)) clean
else
clean-llvm$(1): ;

endif
endef

$(foreach host, $(CFG_HOST), \
 $(eval $(call DEF_CLEAN_LLVM_HOST,$(host))))
