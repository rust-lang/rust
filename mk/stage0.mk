# Extract the snapshot host compiler

$(HBIN0_H_$(CFG_HOST_TRIPLE))/rustc$(X):		\
		$(S)src/snapshots.txt					\
		$(S)src/etc/get-snapshot.py $(MKFILE_DEPS)
	@$(call E, fetch: $@)
	$(Q)$(S)src/etc/get-snapshot.py $(CFG_HOST_TRIPLE)
	$(Q)touch $@

# Host libs will be extracted by the above rule

$(HLIB0_H_$(CFG_HOST_TRIPLE))/$(CFG_RUNTIME): \
		$(HBIN0_H_$(CFG_HOST_TRIPLE))/rustc$(X)
	$(Q)touch $@

## FIXME temporary hack for snapshot transition
CORELIB_DUMMY :=$(call CFG_LIB_NAME,core-dummy)
STDLIB_DUMMY :=$(call CFG_LIB_NAME,std-dummy)

$(HLIB0_H_$(CFG_HOST_TRIPLE))/$(CFG_CORELIB): \
		$(HBIN0_H_$(CFG_HOST_TRIPLE))/rustc$(X)
	$(Q)touch $@
	$(foreach target,$(CFG_TARGET_TRIPLES),\
	$(shell touch $(CFG_HOST_TRIPLE)/stage0/lib/rustc/$(target)/lib/$(CORELIB_DUMMY)))

$(HLIB0_H_$(CFG_HOST_TRIPLE))/$(CFG_STDLIB): \
		$(HBIN0_H_$(CFG_HOST_TRIPLE))/rustc$(X)
	$(Q)touch $@
	$(foreach target,$(CFG_TARGET_TRIPLES),\
	$(shell touch $(CFG_HOST_TRIPLE)/stage0/lib/rustc/$(target)/lib/$(STDLIB_DUMMY)))

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
		$$(TLIB$(2)_T_$(1)_H_$(3))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB0_H_$(1))/$(CFG_CORELIB): \
		$$(TLIB$(2)_T_$(1)_H_$(3))/$$(CFG_CORELIB)
	@$$(call E, cp: $$@)
	$$(Q)cp $$(TLIB$(2)_T_$(1)_H_$(3))/$$(CORELIB_GLOB) $$@

$$(HLIB0_H_$(1))/$(CFG_STDLIB): \
		$$(TLIB$(2)_T_$(1)_H_$(3))/$$(CFG_STDLIB)
	@$$(call E, cp: $$@)
	$$(Q)cp $$(TLIB$(2)_T_$(1)_H_$(3))/$$(STDLIB_GLOB) $$@

$$(HLIB0_H_$(1))/$(CFG_RUSTLLVM): \
		$$(TLIB$(2)_T_$(1)_H_$(3))/$$(CFG_RUSTLLVM)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

# Use stage1 to build other architectures: then you don't have to wait
# for stage2, but you get the latest updates to the compiler source.
$(foreach t,$(NON_HOST_TRIPLES),								\
 $(eval $(call BOOTSTRAP_STAGE0,$(t),1,$(CFG_HOST_TRIPLE))))
