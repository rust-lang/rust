######################################################################
# Cleanup
######################################################################

CLEAN_STAGE_RULES =								\
 $(foreach stage, $(STAGES),					\
  $(foreach host, $(CFG_TARGET_TRIPLES),		\
   clean$(stage)_H_$(host)						\
   $(foreach target, $(CFG_TARGET_TRIPLES),		\
    clean$(stage)_T_$(target)_H_$(host))))

CLEAN_LLVM_RULES = 								\
 $(foreach target, $(CFG_TARGET_TRIPLES),		\
  clean-llvm$(target))

.PHONY: clean clean-all clean-misc

clean-all: clean clean-llvm

clean-llvm: $(CLEAN_LLVM_RULES)

clean: clean-misc $(CLEAN_STAGE_RULES)

clean-misc:
	@$(call E, cleaning)
	$(Q)rm -f $(RUNTIME_OBJS) $(RUNTIME_DEF)
	$(Q)rm -f $(RUSTLLVM_LIB_OBJS) $(RUSTLLVM_OBJS_OBJS) $(RUSTLLVM_DEF)
	$(Q)rm -f $(ML_DEPFILES) $(C_DEPFILES) $(CRATE_DEPFILES)
	$(Q)rm -f $(ML_DEPFILES:%.d=%.d.tmp)
	$(Q)rm -f $(C_DEPFILES:%.d=%.d.tmp)
	$(Q)rm -f $(CRATE_DEPFILES:%.d=%.d.tmp)
	$(Q)rm -Rf $(DOCS)
	$(Q)rm -Rf $(GENERATED)
	$(Q)rm -f rustllvm/$(CFG_RUSTLLVM) rustllvm/rustllvmbits.a
	$(Q)rm -f rt/$(CFG_RUNTIME)
	$(Q)find rustllvm rt -name '*.[odasS]' -delete
	$(Q)find rustllvm rt -name '*.so' -delete
	$(Q)find rustllvm rt -name '*.dylib' -delete
	$(Q)find rustllvm rt -name '*.dll' -delete
	$(Q)find rustllvm rt -name '*.def' -delete
	$(Q)rm -Rf $(wildcard rt/*/libuv/Default)
	$(Q)rm -f test/run_pass_stage2.rc test/run_pass_stage2_driver.rs
	$(Q)rm -Rf $(PKG_NAME)-*.tar.gz dist
	$(Q)rm -Rf $(foreach ext,out out.tmp                      \
                             stage0$(X) stage1$(X) stage2$(X) \
                             bc o s so dll exe dSYM,          \
                        $(wildcard test/*.$(ext) \
                                   test/*/*.$(ext) \
                                   test/bench/*/*.$(ext)))
	$(Q)rm -Rf $(foreach ext, \
                 html aux cp fn ky log pdf pg toc tp vr cps, \
                 $(wildcard doc/*.$(ext) \
                            doc/*/*.$(ext) \
                            doc/*/*/*.$(ext)))
	$(Q)rm -Rf doc/keywords.texi
	$(Q)rm -Rf doc/version.texi
	$(Q)rm -Rf $(foreach sub, index styles files search javascript, \
                 $(wildcard doc/*/$(sub)))
	$(Q)rm -rf libuv

define CLEAN_HOST_STAGE_N

clean$(1)_H_$(2):
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rustc$(X)
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/fuzzer$(X)
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/cargo$(X)
	$(Q)rm -f $$(HBIN$(1)_H_$(2))/rustdoc$(X)
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_RUNTIME)
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_CORELIB)
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_STDLIB)
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_LIBRUSTC)
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CORELIB_GLOB)
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(STDLIB_GLOB)
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(LIBRUSTC_GLOB)
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/$(CFG_RUSTLLVM)
	$(Q)rm -f $$(HLIB$(1)_H_$(2))/libstd.rlib

endef

$(foreach host, $(CFG_TARGET_TRIPLES), \
 $(eval $(foreach stage, $(STAGES), \
  $(eval $(call CLEAN_HOST_STAGE_N,$(stage),$(host))))))

define CLEAN_TARGET_STAGE_N

clean$(1)_T_$(2)_H_$(3):
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rustc$(X)
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/fuzzer$(X)
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/cargo$(X)
	$(Q)rm -f $$(TBIN$(1)_T_$(2)_H_$(3))/rustdoc$(X)
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_RUNTIME)
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_CORELIB)
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_STDLIB)
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_LIBRUSTC)
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CORELIB_GLOB)
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(STDLIB_GLOB)
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(LIBRUSTC_GLOB)
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/$(CFG_RUSTLLVM)
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/libstd.rlib
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/intrinsics.bc
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/intrinsics.ll
	$(Q)rm -f $$(TLIB$(1)_T_$(2)_H_$(3))/libmorestack.a
endef

$(foreach host, $(CFG_TARGET_TRIPLES), \
 $(eval $(foreach target, $(CFG_TARGET_TRIPLES), \
  $(eval $(foreach stage, 0 1 2 3, \
   $(eval $(call CLEAN_TARGET_STAGE_N,$(stage),$(target),$(host))))))))

define DEF_CLEAN_LLVM_TARGET
ifeq ($(CFG_LLVM_ROOT),)
clean-llvm$(1):
	$$(Q)$$(MAKE) -C $$(CFG_LLVM_BUILD_DIR_$(1)) clean
else
clean-llvm$(1): ;

endif
endef

$(foreach target, $(CFG_TARGET_TRIPLES), \
 $(eval $(call DEF_CLEAN_LLVM_TARGET,$(target))))
