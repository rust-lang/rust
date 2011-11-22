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

# For other targets, let the host build the target:

define BOOTSTRAP_STAGE0
  # $(1) target to bootstrap
  # $(2) stage to bootstrap from
  # $(3) target to bootstrap from

$$(HBIN0_H_$(1))/rustc$$(X):								\
		$$(TBIN$(2)_T_$(1)_H_$(3))/rustc$$(X)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB0_H_$(1))/$$(CFG_RUNTIME): \
		$$(TBIN$(2)_T_$(1)_H_$(3))/rustc$$(X)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB0_H_$(1))/$(CFG_STDLIB): \
		$$(TBIN$(2)_T_$(1)_H_$(3))/rustc$$(X)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB0_H_$(1))/$(CFG_RUSTLLVM): \
		$$(TBIN$(2)_T_$(1)_H_$(3))/rustc$$(X)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

$(foreach t,$(NON_HOST_TRIPLES),								\
 $(eval $(call BOOTSTRAP_STAGE0,$(t),2,$(CFG_HOST_TRIPLE))))
