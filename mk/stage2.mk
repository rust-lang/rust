stage2/std.o: $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage1/rustc$(X) stage1/$(CFG_STDLIB) stage1/intrinsics.bc \
              $(LREQ) $(MKFILES)
	@$(call E, compile: $@)
	$(STAGE1) -c --shared -o $@ $<

stage2/$(CFG_STDLIB): stage2/std.o stage2/glue.o
	@$(call E, link: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage2/glue.o $(CFG_GCCISH_LINK_FLAGS) -o \
        $@ $< -Lstage2 -Lrt -lrustrt

stage2/librustc.o: $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ1)
	@$(call E, compile: $@)
	$(STAGE1) -c --shared -o $@ $<

stage2/$(CFG_RUSTCLIB): stage2/librustc.o stage2/glue.o
	@$(call E, link: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage2/glue.o $(CFG_GCCISH_LINK_FLAGS) \
	-o $@ $< -Lstage2 -Lrustllvm -Lrt -lrustrt -lrustllvm -lstd

stage2/rustc.o: $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ1)
	@$(call E, compile: $@)
	$(STAGE1) -c -o $@ $<

stage2/glue.o: stage1/rustc$(X) stage1/$(CFG_STDLIB) stage1/intrinsics.bc \
               rustllvm/$(CFG_RUSTLLVM) rt/$(CFG_RUNTIME)
	@$(call E, generate: $@)
	$(STAGE1) -c -o $@ --glue

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

stage2/%$(X): stage2/%.o  $(SREQ1)
	@$(call E, link [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage2/glue.o -o $@ $< \
      -Lstage2 -Lrustllvm -Lrt rt/main.o -lrustrt -lrustllvm -lstd -lm
	@# dsymutil sometimes fails or prints a warning, but the
	@# program still runs.  Since it simplifies debugging other
	@# programs, I\'ll live with the noise.
	-$(Q)$(CFG_DSYMUTIL) $@
