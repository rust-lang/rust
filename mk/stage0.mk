# FIXME: temporary hack: stdlib and rustrt come in the lib/ directory,
# but we want them in the base directory, so we move them out.
stage0/rustc$(X): $(S)src/snapshots.txt $(S)src/etc/get-snapshot.py $(MKFILES)
	@$(call E, fetch: $@)
	$(Q)$(S)src/etc/get-snapshot.py
	$(Q)mv stage0/lib/$(CFG_STDLIB) stage0/$(CFG_STDLIB)
	$(Q)touch $@

# Host libs will be made in the process of making rustc above.

stage0/$(CFG_RUNTIME): stage0/rustc$(X)
	$(Q)touch $@

stage0/$(CFG_STDLIB): stage0/rustc$(X)
	$(Q)touch $@

stage0/$(CFG_RUSTLLVM): stage0/rustc$(X)
	$(Q)touch $@

# Instantiate template (in stageN.mk) for building
# target libraries.

SREQpre = stage0/lib/main.o $(MKFILES)
$(eval $(call TARGET_LIBS,pre,0,$(CFG_HOST_TRIPLE)))

