$(HOST_BIN0)/rustc$(X): $(S)src/snapshots.txt $(S)src/etc/get-snapshot.py $(MKFILES)
	@$(call E, fetch: $@)
	$(Q)$(S)src/etc/get-snapshot.py
	$(Q)touch $@

# Host libs will be made in the process of making rustc above.

$(HOST_LIB0)/$(CFG_RUNTIME): $(HOST_BIN0)/rustc$(X)
	$(Q)touch $@

$(HOST_LIB0)/$(CFG_STDLIB): $(HOST_BIN0)/rustc$(X)
	$(Q)touch $@

$(HOST_LIB0)/$(CFG_RUSTLLVM): $(HOST_BIN0)/rustc$(X)
	$(Q)touch $@

# Instantiate template (in stageN.mk) for building
# target libraries.

SREQpre = $(MKFILES)
$(eval $(call TARGET_LIBS,pre,0,$(CFG_HOST_TRIPLE)))

