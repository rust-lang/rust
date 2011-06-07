stage3/std.o: $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage2/rustc$(X) stage2/$(CFG_STDLIB) stage2/intrinsics.bc \
              $(LREQ) $(MKFILES)
	@$(call E, compile: $@)
	$(STAGE2) -c --shared -o $@ $<

stage3/$(CFG_STDLIB): stage3/std.o stage3/glue.o
	@$(call E, link: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage3/glue.o $(CFG_GCCISH_LINK_FLAGS) -o \
        $@ $< -Lstage3 -Lrt -lrustrt

stage3/librustc.o: $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ2)
	@$(call E, compile: $@)
	$(STAGE2) -c --shared -o $@ $<

stage3/$(CFG_RUSTCLIB): stage3/librustc.o stage3/glue.o
	@$(call E, link: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage3/glue.o $(CFG_GCCISH_LINK_FLAGS) \
	-o $@ $< -Lstage3 -Lrustllvm -Lrt -lrustrt -lrustllvm -lstd

stage3/rustc.o: $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ2)
	@$(call E, compile: $@)
	$(STAGE2) -c -o $@ $<

stage3/glue.o: stage2/rustc$(X) stage2/$(CFG_STDLIB) stage2/intrinsics.bc \
               rustllvm/$(CFG_RUSTLLVM) rt/$(CFG_RUNTIME)
	@$(call E, generate: $@)
	$(STAGE2) -c -o $@ --glue

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

stage3/%$(X): stage3/%.o  $(SREQ2)
	@$(call E, link [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage3/glue.o -o $@ $< \
      -Lstage3 -Lrustllvm -Lrt rt/main.a -lrustrt -lrustllvm -lstd -lm
	@# dsymutil sometimes fails or prints a warning, but the
	@# program still runs.  Since it simplifies debugging other
	@# programs, I\'ll live with the noise.
	-$(Q)$(CFG_DSYMUTIL) $@
