stage0/bin/rustc$(X): $(S)src/snapshots.txt $(S)src/etc/get-snapshot.py $(MKFILES)
	@$(call E, fetch: $@)
	$(Q)$(S)src/etc/get-snapshot.py
	$(Q)touch $@

# Host libs will be made in the process of making rustc above.

stage0/lib/$(CFG_RUNTIME): stage0/bin/rustc$(X)
	$(Q)touch $@

stage0/lib/$(CFG_STDLIB): stage0/bin/rustc$(X)
	$(Q)touch $@

stage0/lib/$(CFG_RUSTLLVM): stage0/bin/rustc$(X)
	$(Q)touch $@

# Instantiate template (in stageN.mk) for building
# target libraries.

SREQpre = stage0/lib/$(CFG_HOST_TRIPLE)/main.o $(MKFILES)
$(eval $(call TARGET_LIBS,pre,0,$(CFG_HOST_TRIPLE)))

