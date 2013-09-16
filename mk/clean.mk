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

CLEAN_STAGE_RULES =								\
 $(foreach stage, $(STAGES),					\
  $(foreach host, $(CFG_HOST_TRIPLES),		\
   clean$(stage)_H_$(host)						\
   $(foreach target, $(CFG_TARGET_TRIPLES),		\
    clean$(stage)_T_$(target)_H_$(host))))

CLEAN_LLVM_RULES = 								\
 $(foreach target, $(CFG_HOST_TRIPLES),		\
  clean-llvm$(target))

.PHONY: clean clean-all clean-misc clean-llvm

clean-all: clean clean-llvm

clean-llvm: $(CLEAN_LLVM_RULES)

clean: clean-misc $(CLEAN_STAGE_RULES)

clean-misc:
	@$(call E, cleaning)
	$(Q)find $(CFG_BUILD_TRIPLE)/rustllvm \
	         $(CFG_BUILD_TRIPLE)/rt \
		 $(CFG_BUILD_TRIPLE)/test \
         -name '*.[odasS]' -o \
         -name '*.so' -o      \
         -name '*.dylib' -o   \
         -name '*.dll' -o     \
         -name '*.def' -o     \
         -name '*.bc'         \
         | xargs rm -f
	$(Q)find $(CFG_BUILD_TRIPLE)\
         -name '*.dSYM'       \
         | xargs rm -Rf
	$(Q)rm -f $(RUNTIME_OBJS) $(RUNTIME_DEF)
	$(Q)rm -f $(RUSTLLVM_LIB_OBJS) $(RUSTLLVM_OBJS_OBJS) $(RUSTLLVM_DEF)
	$(Q)rm -Rf $(DOCS)
	$(Q)rm -Rf $(GENERATED)
	$(Q)rm -Rf tmp/*
	$(Q)rm -Rf rust-stage0-*.tar.bz2 $(PKG_NAME)-*.tar.gz dist
	$(Q)rm -Rf $(foreach ext, \
                 html aux cp fn ky log pdf pg toc tp vr cps, \
                 $(wildcard doc/*.$(ext) \
                            doc/*/*.$(ext) \
                            doc/*/*/*.$(ext)))
	$(Q)rm -Rf doc/version.md
	$(Q)rm -Rf $(foreach sub, index styles files search javascript, \
                 $(wildcard doc/*/$(sub)))

define CLEAN_HOST_STAGE_N

clean$(1)_H_$(2):
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rustc$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rustpkg$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/serializer$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rustdoc$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rustdoc_ng$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rusti$(X_$(2))
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rust$(X_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTPKG_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTDOC_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTDOCNG_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_RUNTIME_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_STDLIB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_EXTRALIB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTC_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBSYNTAX_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTI_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUST_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(STDLIB_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(EXTRALIB_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTC_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBSYNTAX_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTPKG_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTDOC_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTDOCNG_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTI_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUST_GLOB_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_RUSTLLVM_$(2))
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/libstd.rlib

endef

$(foreach host, $(CFG_HOST_TRIPLES), \
 $(eval $(foreach stage, $(STAGES), \
  $(eval $(call CLEAN_HOST_STAGE_N,$(stage),$(host))))))

define CLEAN_TARGET_STAGE_N

clean$(1)_T_$(2)_H_$(3):
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rustc$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rustpkg$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/serializer$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rustdoc$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rustdoc_ng$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rusti$(X_$(2))
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rust$(X_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTPKG_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTDOC_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTDOCNG_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_RUNTIME_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_STDLIB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_EXTRALIB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTC_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBSYNTAX_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTI_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUST_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(STDLIB_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(EXTRALIB_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTC_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBSYNTAX_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTPKG_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTDOC_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTDOCNG_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTI_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUST_GLOB_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_RUSTLLVM_$(2))
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/libstd.rlib
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/libmorestack.a
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/librun_pass_stage* # For unix
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/run_pass_stage* # For windows
endef

$(foreach host, $(CFG_HOST_TRIPLES), \
 $(eval $(foreach target, $(CFG_TARGET_TRIPLES), \
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

$(foreach host, $(CFG_HOST_TRIPLES), \
 $(eval $(call DEF_CLEAN_LLVM_HOST,$(host))))
