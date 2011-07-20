# FIXME: temporary hack: stdlib comes in the lib/ directory, but we want it in
# the base directory, so we move it out.
stage0/rustc$(X): $(S)src/snapshots.txt $(S)src/etc/get-snapshot.py $(MKFILES)
	@$(call E, fetch: $@)
	$(Q)$(S)src/etc/get-snapshot.py
	$(Q)mv stage0/lib/$(CFG_STDLIB) stage0/$(CFG_STDLIB)
	$(Q)touch $@

# Host libs will be made in the process of making rustc above.

# FIXME: temporary hack: the runtime is currently carried in
# lib/ directory only, so we copy it out.

stage0/$(CFG_RUNTIME): stage0/lib/$(CFG_RUNTIME)
	$(Q)cp $< $@

stage0/$(CFG_STDLIB): stage0/rustc$(X)
	$(Q)touch $@

stage0/$(CFG_RUSTLLVM): stage0/rustc$(X)
	$(Q)touch $@

# Target libs will be made in the process of making rustc above.

stage0/lib/glue.o: stage0/rustc$(X)
	$(Q)touch $@

# FIXME: temporary hack: currently not distributing main.o like we should;
# copying from rt

stage0/lib/main.o: rt/main.o
	$(Q)cp $< $@


stage0/lib/$(CFG_RUNTIME): stage0/rustc$(X)
	$(Q)touch $@

# stage0/lib/$(CFG_STDLIB) and stage0/lib/libstd.rlib rules are generated
# in stageN.mk

