# Extract the snapshot host compiler

$(HBIN0_H_$(CFG_HOST_TRIPLE))/rustc$(X):		\
		$(S)src/snapshots.txt					\
		$(S)src/etc/get-snapshot.py $(MKFILES)
	@$(call E, fetch: $@)
	$(Q)$(S)src/etc/get-snapshot.py $(CFG_HOST_TRIPLE)
	$(Q)touch $@

# Host libs will be extracted by the above rule

$(HLIB0_H_$(CFG_HOST_TRIPLE))/$(CFG_RUNTIME): \
		$(HBIN0_H_$(CFG_HOST_TRIPLE))/rustc$(X)
	$(Q)touch $@

$(HLIB0_H_$(CFG_HOST_TRIPLE))/$(CFG_STDLIB): \
		$(HBIN0_H_$(CFG_HOST_TRIPLE))/rustc$(X)
	$(Q)touch $@

$(HLIB0_H_$(CFG_HOST_TRIPLE))/$(CFG_RUSTLLVM): \
		$(HBIN0_H_$(CFG_HOST_TRIPLE))/rustc$(X)
	$(Q)touch $@
