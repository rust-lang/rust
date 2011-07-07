stage2/lib/$(CFG_STDLIB): $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage2/rustc$(X) stage1/lib/$(CFG_STDLIB) stage2/intrinsics.bc \
              stage2/lib/$(CFG_RUNTIME) stage2/lib/$(CFG_RUSTLLVM) \
              stage2/lib/glue.o $(LREQ) $(MKFILES)
	@$(call E, compile_and_link: $@)
	$(STAGE2)  --lib -o $@ $<

stage2/lib/libstd.rlib:  $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage2/rustc$(X) stage1/lib/$(CFG_STDLIB) stage2/intrinsics.bc \
              stage2/lib/glue.o $(LREQ) $(MKFILES)
	@$(call E, compile_and_link: $@)
	$(STAGE2) --lib --static -o $@ $<

stage2/lib/glue.o: stage2/rustc$(X) stage1/lib/$(CFG_STDLIB) \
	stage2/intrinsics.bc rustllvm/$(CFG_RUSTLLVM) rt/$(CFG_RUNTIME)
	@$(call E, generate: $@)
	$(STAGE2) -c -o $@ --glue

stage2/intrinsics.bc:	$(INTRINSICS_BC)
	@$(call E, cp: $@)
	$(Q)cp $< $@

stage2/lib/$(CFG_RUNTIME):	rt/$(CFG_RUNTIME)
	@$(call E, cp: $@)
	$(Q)cp $< $@

stage2/lib/$(CFG_RUSTLLVM):	rustllvm/$(CFG_RUSTLLVM)
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

stage2/%$(X): $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ1) stage1/intrinsics.bc
	@$(call E, compile_and_link: $@)
	$(STAGE1) -o $@ $<
