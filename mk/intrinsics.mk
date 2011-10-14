######################################################################
# intrinsics.bc rules
######################################################################

# TODO: Use clang to compile the C++.
INTRINSICS_LL_IN := $(S)src/rt/intrinsics/intrinsics.ll.in
INTRINSICS_LL := intrinsics/intrinsics.ll
INTRINSICS_BC := intrinsics/intrinsics.bc

$(INTRINSICS_LL):  $(INTRINSICS_LL_IN) $(MKFILES)
	@$(call E, mkdir: intrinsics)
	$(Q)mkdir -p intrinsics
	@$(call E, sed: $@)
	$(Q)sed s/@CFG_TARGET_TRIPLE@/$(CFG_LLVM_TRIPLE)/g $< > $@

$(INTRINSICS_BC):   $(INTRINSICS_LL) $(MKFILES)
	@$(call E, llvm-as: $@)
	$(Q)$(LLVM_AS) -o $@ $<

