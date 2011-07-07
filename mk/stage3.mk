stage3/lib/$(CFG_STDLIB): $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage3/rustc$(X) stage2/lib/$(CFG_STDLIB) stage3/intrinsics.bc \
              stage3/lib/$(CFG_RUNTIME) stage3/lib/$(CFG_RUSTLLVM) \
              stage3/glue.o $(LREQ) $(MKFILES)
	@$(call E, compile_and_link: $@)
	$(STAGE3)  --lib -o $@ $<

stage3/lib/libstd.rlib:  $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage3/rustc$(X) stage2/lib/$(CFG_STDLIB) stage3/intrinsics.bc \
              stage3/glue.o $(LREQ) $(MKFILES)
	@$(call E, compile_and_link: $@)
	$(STAGE3) --lib --static -o $@ $<

stage3/lib/glue.o: stage3/rustc$(X) stage2/lib/$(CFG_STDLIB) \
	 stage3/intrinsics.bc rustllvm/$(CFG_RUSTLLVM) rt/$(CFG_RUNTIME)
	@$(call E, generate: $@)
	$(STAGE3) -c -o $@ --glue

stage3/glue.o: stage3/lib/glue.o
	cp stage3/lib/glue.o stage3/glue.o

stage3/intrinsics.bc:	$(INTRINSICS_BC)
	@$(call E, cp: $@)
	$(Q)cp $< $@

stage3/lib/$(CFG_RUNTIME):	rt/$(CFG_RUNTIME)
	@$(call E, cp: $@)
	$(Q)cp $< $@

stage3/lib/$(CFG_RUSTLLVM):	rustllvm/$(CFG_RUSTLLVM)
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

stage3/%$(X): $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ2) stage2/intrinsics.bc
	@$(call E, compile_and_link: $@)
	$(STAGE2) -o $@ $<
