######################################################################
# Cleanup
######################################################################

CLEAN_STAGE_RULES = $(foreach target,$(CFG_TARGET_TRIPLES), \
 clean0$(target) clean1$(target) clean2$(target) clean3$(target)) \
 clean0 clean1 clean2 clean3

CLEAN_LLVM_RULES = $(foreach target,$(CFG_TARGET_TRIPLES), \
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
	$(Q)rm -f $(GENERATED)
	$(Q)rm -f rustllvm/$(CFG_RUSTLLVM) rustllvm/rustllvmbits.a
	$(Q)rm -f rt/$(CFG_RUNTIME)
	$(Q)find rt -name '*.o' -delete
	$(Q)find rt -name '*.a' -delete
	$(Q)rm -f test/run_pass_stage2.rc test/run_pass_stage2_driver.rs
	$(Q)rm -Rf $(PKG_NAME)-*.tar.gz dist
	$(Q)rm -f $(foreach ext,o a d bc s exe,$(wildcard stage*/*.$(ext)))
	$(Q)rm -Rf $(foreach ext,out out.tmp                      \
                             stage0$(X) stage1$(X) stage2$(X) \
                             bc o s exe dSYM,                 \
                        $(wildcard test/*.$(ext) \
                                   test/*/*.$(ext) \
                                   test/bench/*/*.$(ext)))
	$(Q)rm -Rf $(foreach ext, \
                 aux cp fn ky log pdf html pg toc tp vr cps, \
                 $(wildcard doc/*.$(ext)))
	$(Q)rm -Rf doc/version.texi
	$(Q)rm -rf libuv

define CLEAN_STAGE_N

clean$(1):
	$(Q)rm -f $$(HOST_BIN$(1))/rustc$(X)
	$(Q)rm -f $$(HOST_BIN$(1))/fuzzer$(X)
	$(Q)rm -f $$(HOST_LIB$(1))/$(CFG_RUNTIME)
	$(Q)rm -f $$(HOST_LIB$(1))/$(CFG_STDLIB)
	$(Q)rm -f $$(HOST_LIB$(1))/$(CFG_RUSTLLVM)
	$(Q)rm -f $$(HOST_LIB$(1))/libstd.rlib

clean$(1)$(2):
	$(Q)rm -f $$(TARGET_BIN$(1)$(2))/rustc$(X)
	$(Q)rm -f $$(TARGET_BIN$(1)$(2))/fuzzer$(X)
	$(Q)rm -f $$(TARGET_LIB$(1)$(2))/$(CFG_RUNTIME)
	$(Q)rm -f $$(TARGET_LIB$(1)$(2))/$(CFG_STDLIB)
	$(Q)rm -f $$(TARGET_LIB$(1)$(2))/$(CFG_RUSTLLVM)
	$(Q)rm -f $$(TARGET_LIB$(1)$(2))/libstd.rlib
	$(Q)rm -f $$(TARGET_LIB$(1)$(2))/intrinsics.bc

endef

$(foreach target, $(CFG_TARGET_TRIPLES), \
 $(eval $(call CLEAN_STAGE_N,0,$(target))) \
 $(eval $(call CLEAN_STAGE_N,1,$(target))) \
 $(eval $(call CLEAN_STAGE_N,2,$(target))) \
 $(eval $(call CLEAN_STAGE_N,3,$(target))))


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
