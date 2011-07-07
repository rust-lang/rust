stage0/rustc$(X): $(S)src/snapshots.txt $(S)src/etc/get-snapshot.py $(MKFILES)
	@$(call E, fetch: $@)
	$(Q)$(S)src/etc/get-snapshot.py
	$(Q)touch $@

# These two will be made in the process of making rustc above.

stage0/lib/glue.o: stage0/rustc$(X)
	$(Q)touch $@

stage0/lib/$(CFG_STDLIB): stage0/rustc$(X)
	$(Q)touch $@

# TODO: Include as part of the snapshot.
stage0/intrinsics.bc:   $(INTRINSICS_BC)
	@$(call E, cp: $@)
	$(Q)cp $< $@

