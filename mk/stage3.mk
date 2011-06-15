stage3/$(CFG_STDLIB): $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage3/rustc$(X) stage2/$(CFG_STDLIB) stage3/intrinsics.bc \
              $(LREQ) $(MKFILES)
	@$(call E, compile_and_link: $@)
	$(STAGE3)  --shared -o $@ $<

stage3/glue.o: stage3/rustc$(X) stage2/$(CFG_STDLIB) stage3/intrinsics.bc \
               rustllvm/$(CFG_RUSTLLVM) rt/$(CFG_RUNTIME)
	@$(call E, generate: $@)
	$(STAGE3) -c -o $@ --glue

stage3/intrinsics.bc:	$(INTRINSICS_BC)
	@$(call E, cp: $@)
	$(Q)cp $< $@

# Due to make not wanting to run the same implicit rules twice on the same
# rule tree (implicit-rule recursion prevention, see "Chains of Implicit
# Rules" in GNU Make manual) we have to re-state the %.o and %.s patterns here
# for different directories, to handle cases where (say) a test relies on a
# compiler that relies on a .o file.

stage3/%.o: stage3/%.s
	@$(call E, assemble [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) -o $@ -c $<

stage3/%$(X): $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ2)
	@$(call E, compile_and_link: $@)
	$(STAGE2) -o $@ $<
