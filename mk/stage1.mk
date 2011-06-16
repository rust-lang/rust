stage1/$(CFG_STDLIB): $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage1/rustc$(X) stage0/$(CFG_STDLIB) stage1/intrinsics.bc \
              stage1/glue.o $(LREQ) $(MKFILES)
	@$(call E, compile_and_link: $@)
	$(STAGE1)  --shared -o $@ $<

stage1/glue.o: stage1/rustc$(X) stage0/$(CFG_STDLIB) stage1/intrinsics.bc \
               $(LREQ) $(MKFILES)
	@$(call E, generate: $@)
	$(STAGE1) -c -o $@ --glue

stage1/intrinsics.bc:	$(INTRINSICS_BC)
	@$(call E, cp: $@)
	$(Q)cp $< $@

# Due to make not wanting to run the same implicit rules twice on the same
# rule tree (implicit-rule recursion prevention, see "Chains of Implicit
# Rules" in GNU Make manual) we have to re-state the %.o and %.s patterns here
# for different directories, to handle cases where (say) a test relies on a
# compiler that relies on a .o file.

stage1/%.o: stage1/%.s
	@$(call E, assemble [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) -o $@ -c $<

stage1/%$(X): $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ0) stage0/intrinsics.bc
	@$(call E, compile_and_link: $@)
	$(STAGE0) -o $@ $<
