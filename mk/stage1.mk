stage1/std.o: $(STDLIB_CRATE) $(STDLIB_INPUTS) \
              stage0/rustc$(X) stage0/$(CFG_STDLIB) stage0/intrinsics.bc \
              $(LREQ) $(MKFILES)
	@$(call E, compile: $@)
	$(STAGE0) -c --shared -o $@ $<

stage1/$(CFG_STDLIB): stage1/std.o stage1/glue.o
	@$(call E, link: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage1/glue.o $(CFG_GCCISH_LINK_FLAGS) \
        -o $@ $< -Lstage1 -Lrt -lrustrt

stage1/librustc.o: $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ0)
	@$(call E, compile: $@)
	$(STAGE0) -c --shared -o $@ $<

stage1/$(CFG_RUSTCLIB): stage1/librustc.o stage1/glue.o
	@$(call E, link: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage1/glue.o $(CFG_GCCISH_LINK_FLAGS) \
	-o $@ $< -Lstage1 -Lrustllvm -Lrt -lrustrt -lrustllvm -lstd

stage1/rustc.o: $(COMPILER_CRATE) $(COMPILER_INPUTS) $(SREQ0)
	@$(call E, compile: $@)
	$(STAGE0) -c -o $@ $<

stage1/glue.o: stage0/rustc$(X) stage0/$(CFG_STDLIB) stage0/intrinsics.bc \
               $(LREQ) $(MKFILES)
	@$(call E, generate: $@)
	$(STAGE0) -c -o $@ --glue

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

stage1/%$(X): stage1/%.o  $(SREQ0)
	@$(call E, link [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage1/glue.o -o $@ $< \
      -Lstage1 -Lrustllvm -Lrt rt/main.a -lrustrt -lrustllvm -lstd -lm
	@# dsymutil sometimes fails or prints a warning, but the
	@# program still runs.  Since it simplifies debugging other
	@# programs, I\'ll live with the noise.
	-$(Q)$(CFG_DSYMUTIL) $@
