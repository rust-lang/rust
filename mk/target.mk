# TARGET_STAGE_N template: This defines how target artifacts are built
# for all stage/target architecture combinations. The arguments:
# $(1) is the stage
# $(2) is the target triple
# $(3) is the host triple

define TARGET_STAGE_N

$$(TLIB$(1)_T_$(2)_H_$(3))/intrinsics.ll: \
		$$(S)src/rt/intrinsics/intrinsics.$(HOST_$(2)).ll.in
	@$$(call E, sed: $$@)
	$$(Q)sed s/@CFG_TARGET_TRIPLE@/$(2)/ $$< > $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/intrinsics.bc: \
		$$(TLIB$(1)_T_$(2)_H_$(3))/intrinsics.ll \
		$$(LLVM_CONFIG_$(2))
	@$$(call E, llvms-as: $$@)
	$$(Q)$$(LLVM_AS_$(2)) -o $$@ $$<

$$(TLIB$(1)_T_$(2)_H_$(3))/libmorestack.a: \
		rt/$(2)/arch/$$(HOST_$(2))/libmorestack.a
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_CORELIB): \
		$$(CORELIB_CRATE) $$(CORELIB_INPUTS) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) --lib -o $$@ $$<

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_STDLIB): \
		$$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
        $$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_CORELIB) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) --lib -o $$@ $$<

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUNTIME): \
		rt/$(2)/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUSTLLVM): \
		rustllvm/$(2)/$$(CFG_RUSTLLVM)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TBIN$(1)_T_$(2)_H_$(3))/rustc$$(X):				\
		$$(COMPILER_CRATE) $$(COMPILER_INPUTS)		\
		$$(TSREQ$(1)_T_$(2)_H_$(3))					\
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUSTLLVM)	\
		$$(TSTDLIB_DEFAULT$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3))  -o $$@ $$<

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBRUSTC):		\
		$$(COMPILER_CRATE) $$(COMPILER_INPUTS)		\
		$$(TSREQ$(1)_T_$(2)_H_$(3))					\
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUSTLLVM)	\
		$$(TSTDLIB_DEFAULT$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3))  --lib -o $$@ $$<

endef

# In principle, each host can build each target:
$(foreach source,$(CFG_TARGET_TRIPLES),						\
 $(foreach target,$(CFG_TARGET_TRIPLES),					\
  $(eval $(call TARGET_STAGE_N,0,$(source),$(target)))		\
  $(eval $(call TARGET_STAGE_N,1,$(source),$(target)))		\
  $(eval $(call TARGET_STAGE_N,2,$(source),$(target)))		\
  $(eval $(call TARGET_STAGE_N,3,$(source),$(target)))))
