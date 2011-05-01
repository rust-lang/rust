stage0/$(CFG_STDLIB): $(STDLIB_CRATE) $(STDLIB_INPUTS) \
                      boot/rustboot$(X) $(MKFILES)
	@$(call E, compile: $@)
	$(BOOT) -shared -o $@ $<

stage0/rustc$(X): $(COMPILER_CRATE) $(COMPILER_INPUTS) $(BREQ)
	@$(call E, compile: $@)
	$(BOOT) -minimal -o $@ $<
	$(Q)chmod 0755 $@
