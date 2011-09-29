######################################################################
# Cleanup
######################################################################

CLEAN_STAGE_RULES = $(foreach target,$(CFG_TARGET_TRIPLES), \
 clean0$(target) clean1$(target) clean2$(target) clean3$(target))


.PHONY: clean

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
	$(Q)rm -f stage0/rustc$(X) stage0/lib/main.o
	$(Q)rm -f stage0/lib/$(CFG_RUNTIME) stage0/lib/$(CFG_STDLIB)
	$(Q)rm -f stage0/$(CFG_RUNTIME) stage0/$(CFG_STDLIB)
	$(Q)rm -f stage0/lib/libstd.rlib
	$(Q)rm -f stage0/$(CFG_RUSTLLVM) stage0/lib/intrinsics.bc
	$(Q)rm -f stage1/rustc$(X) stage1/lib/main.o
	$(Q)rm -f stage1/lib/$(CFG_RUNTIME) stage1/lib/$(CFG_STDLIB)
	$(Q)rm -f stage1/$(CFG_RUNTIME) stage1/$(CFG_STDLIB)
	$(Q)rm -f stage1/$(CFG_RUSTLLVM) stage1/lib/intrinsics.bc
	$(Q)rm -f stage1/lib/libstd.rlib
	$(Q)rm -f stage2/rustc$(X) stage2/lib/main.o
	$(Q)rm -f stage2/lib/$(CFG_RUNTIME) stage2/lib/$(CFG_STDLIB)
	$(Q)rm -f stage2/$(CFG_RUNTIME) stage2/$(CFG_STDLIB)
	$(Q)rm -f stage2/$(CFG_RUSTLLVM) stage2/lib/intrinsics.bc
	$(Q)rm -f stage2/lib/libstd.rlib
	$(Q)rm -f stage3/rustc$(X) stage3/lib/main.o
	$(Q)rm -f stage3/lib/$(CFG_RUNTIME) stage3/lib/$(CFG_STDLIB)
	$(Q)rm -f stage3/$(CFG_RUNTIME) stage3/$(CFG_STDLIB)
	$(Q)rm -f stage3/$(CFG_RUSTLLVM) stage3/lib/intrinsics.bc
	$(Q)rm -f stage3/lib/libstd.rlib
	$(Q)rm -f stage1/fuzzer stage1/lib/$(CFG_LIBRUSTC)
	$(Q)rm -f rustllvm/$(CFG_RUSTLLVM) rustllvm/rustllvmbits.a
	$(Q)rm -f rt/$(CFG_RUNTIME)
	$(Q)rm -f rt/main.o
	$(Q)rm -f rt/main.ll
	$(Q)rm -f rt/libuv/uv.a
	$(Q)rm -Rf $(wildcard rt/libuv/src/*/*)
	$(Q)rm -f $(wildcard rt/libuv/src/*.o)
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
	$(Q)rm -rf rt/libuv

define CLEAN_STAGE_N

clean$(1)$(2):
	$(Q)rm -f stage$(1)/bin/rustc
	$(Q)rm -f stage$(1)/bin/fuzzer
	$(Q)rm -f stage$(1)/lib/$(CFG_RUNTIME)
	$(Q)rm -f stage$(1)/lib/$(CFG_STDLIB)
	$(Q)rm -f stage$(1)/lib/$(CFG_RUSTLLVM)
	$(Q)rm -f stage$(1)/lib/rustc/$(2)/$(CFG_RUNTIME)
	$(Q)rm -f stage$(1)/lib/rustc/$(2)/$(CFG_STDLIB)
	$(Q)rm -f stage$(1)/lib/rustc/$(2)/libstd.rlib
	$(Q)rm -f stage$(1)/lib/rustc/$(2)/intrinsics.bc
	$(Q)rm -f stage$(1)/lib/rustc/$(2)/main.o

endef

$(foreach target, $(CFG_TARGET_TRIPLES), \
 $(eval $(call CLEAN_STAGE_N,0,$(target))) \
 $(eval $(call CLEAN_STAGE_N,1,$(target))) \
 $(eval $(call CLEAN_STAGE_N,2,$(target))) \
 $(eval $(call CLEAN_STAGE_N,3,$(target))))
