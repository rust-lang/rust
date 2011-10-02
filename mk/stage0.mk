# Extract the snapshot host compiler

$(HOST_BIN0)/rustc$(X): \
	$(S)src/snapshots.txt \
	$(S)src/etc/get-snapshot.py $(MKFILES)
	@$(call E, fetch: $@)
	$(Q)$(S)src/etc/get-snapshot.py
	$(Q)touch $@

# Host libs will be extracted by the above rule

$(HOST_LIB0)/$(CFG_RUNTIME): $(HOST_BIN0)/rustc$(X)
	$(Q)touch $@

$(HOST_LIB0)/$(CFG_STDLIB): $(HOST_BIN0)/rustc$(X)
	$(Q)touch $@

$(HOST_LIB0)/$(CFG_RUSTLLVM): $(HOST_BIN0)/rustc$(X)
	$(Q)touch $@
