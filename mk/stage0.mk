# Extract the snapshot host compiler

$(HBIN0_H_$(CFG_BUILD))/:
	mkdir -p $@

$(HLIB0_H_$(CFG_BUILD))/:
	mkdir -p $@

$(SNAPSHOT_RUSTC_POST_CLEANUP):						\
		$(S)src/snapshots.txt					\
		$(S)src/etc/get-snapshot.py $(MKFILE_DEPS)		\
		| $(HBIN0_H_$(CFG_BUILD))/

	@$(call E, fetch: $@)
#   Note: the variable "SNAPSHOT_FILE" is generally not set, and so
#   we generally only pass one argument to this script.
ifdef CFG_ENABLE_LOCAL_RUST
	$(Q)$(S)src/etc/local_stage0.sh $(CFG_BUILD) $(CFG_LOCAL_RUST_ROOT)
else
	$(Q)$(CFG_PYTHON) $(S)src/etc/get-snapshot.py $(CFG_BUILD) $(SNAPSHOT_FILE)
ifdef CFG_ENABLE_PAX_FLAGS
	@$(call E, apply PaX flags: $@)
	@"$(CFG_PAXCTL)" -cm "$@"
endif
endif
	$(Q)touch $@

# For other targets, let the host build the target:

define BOOTSTRAP_STAGE0
  # $(1) target to bootstrap
  # $(2) stage to bootstrap from
  # $(3) target to bootstrap from

$(HBIN0_H_$(1))/:
	mkdir -p $@

$(HLIB0_H_$(1))/:
	mkdir -p $@

$$(HBIN0_H_$(1))/rustc$$(X_$(1)): \
		$$(TBIN$(2)_T_$(1)_H_$(3))/rustc$$(X_$(1)) \
		| $(HBIN0_H_$(1))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

# Use stage1 to build other architectures: then you don't have to wait
# for stage2, but you get the latest updates to the compiler source.
$(foreach t,$(NON_BUILD_HOST),								\
 $(eval $(call BOOTSTRAP_STAGE0,$(t),1,$(CFG_BUILD))))
