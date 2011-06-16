stage2/$(CFG_STDLIB): $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage2/rustc$(X) stage1/$(CFG_STDLIB) stage2/intrinsics.bc \
              stage2/glue.o $(LREQ) $(MKFILES)
	@$(call E, compile_and_link: $@)
	$(STAGE2)  --shared -o $@ $<

stage2/glue.o: stage2/rustc$(X) stage1/$(CFG_STDLIB) stage1/intrinsics.bc \
               rustllvm/$(CFG_RUSTLLVM) rt/$(CFG_RUNTIME)
	@$(call E, generate: $@)
	$(STAGE2) -c -o $@ --glue

stage2/intrinsics.bc:	$(INTRINSICS_BC)
	@$(call E, cp: $@)
	$(Q)cp $< $@

# Due to make not wanting to run the same implicit rules twice on the same
# rule tree (implicit-rule recursion prevention, see "Chains of Implicit
# Rules" in GNU Make manual) we have to re-state the %.o and %.s patterns here
# for different directories, to handle cases where (say) a test relies on a
# compiler that relies on a .o file.

stage2/%.o: stage2/%.s
	@$(call E, assemble [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) -o $@ -c $<

stage2/%$(X): $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ1)
	@$(call E, compile_and_link: $@)
	$(STAGE1) -o $@ $<
