stage0/rustc$(X): $(S)src/snapshots.txt $(S)src/etc/get-snapshot.py $(MKFILES)
	@$(call E, fetch: $@)
	$(S)src/etc/get-snapshot.py

# These two will be made in the process of making rustc above.

stage0/glue.o: stage0/rustc$(X)

stage0/$(CFG_STDLIB): stage0/rustc$(X)
