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

# Host libs will be extracted by the above rule

# NOTE: remove all these after the next snapshot
$(HLIB0_H_$(CFG_BUILD))/$(CFG_STDLIB_$(CFG_BUILD)): \
		$(HBIN0_H_$(CFG_BUILD))/rustc$(X_$(CFG_BUILD)) \
		| $(HLIB0_H_$(CFG_BUILD))/
	$(Q)touch $@

$(HLIB0_H_$(CFG_BUILD))/$(CFG_EXTRALIB_$(CFG_BUILD)): \
		$(HBIN0_H_$(CFG_BUILD))/rustc$(X_$(CFG_BUILD)) \
		| $(HLIB0_H_$(CFG_BUILD))/
	$(Q)touch $@

$(HLIB0_H_$(CFG_BUILD))/$(CFG_LIBRUSTUV_$(CFG_BUILD)): \
		$(HBIN0_H_$(CFG_BUILD))/rustc$(X_$(CFG_BUILD)) \
		| $(HLIB0_H_$(CFG_BUILD))/
	$(Q)touch $@

$(HLIB0_H_$(CFG_BUILD))/$(CFG_LIBRUSTC_$(CFG_BUILD)): \
		$(HBIN0_H_$(CFG_BUILD))/rustc$(X_$(CFG_BUILD)) \
		| $(HLIB0_H_$(CFG_BUILD))/
	$(Q)touch $@

$(HLIB0_H_$(CFG_BUILD))/$(CFG_RUSTLLVM_$(CFG_BUILD)): \
		$(HBIN0_H_$(CFG_BUILD))/rustc$(X_$(CFG_BUILD)) \
		| $(HLIB0_H_$(CFG_BUILD))/
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

# NOTE: removing everything below after the next snapshot
$$(HLIB0_H_$(1))/$(CFG_STDLIB_$(1)): \
		$$(TLIB$(2)_T_$(1)_H_$(3))/$(CFG_STDLIB_$(1)) \
		| $(HLIB0_H_$(1))/
	@$$(call E, cp: $$@)
	$$(call CHECK_FOR_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$(TLIB$(2)_T_$(1)_H_$(3))/$(STDLIB_GLOB_$(1)) $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(STDLIB_GLOB_$(4)),$$(notdir $$@))

$$(HLIB0_H_$(1))/$(CFG_EXTRALIB_$(1)): \
		$$(TLIB$(2)_T_$(1)_H_$(3))/$(CFG_EXTRALIB_$(1)) \
		| $(HLIB0_H_$(1))/
	@$$(call E, cp: $$@)
	$$(call CHECK_FOR_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$(TLIB$(2)_T_$(1)_H_$(3))/$(EXTRALIB_GLOB_$(1)) $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(EXTRALIB_GLOB_$(4)),$$(notdir $$@))

$$(HLIB0_H_$(1))/$(CFG_LIBRUSTUV_$(1)): \
		$$(TLIB$(2)_T_$(1)_H_$(3))/$(CFG_LIBRUSTUV_$(1)) \
		| $(HLIB0_H_$(1))/
	@$$(call E, cp: $$@)
	$$(call CHECK_FOR_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$(TLIB$(2)_T_$(1)_H_$(3))/$(LIBRUSTUV_GLOB_$(1)) $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTUV_GLOB_$(4)),$$(notdir $$@))

$$(HLIB0_H_$(1))/$(CFG_LIBRUSTC_$(1)): \
		$$(TLIB$(2)_T_$(1)_H_$(3))/$(CFG_LIBRUSTC_$(1)) \
		| $(HLIB0_H_$(1))/
	@$$(call E, cp: $$@)
	$$(call CHECK_FOR_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTC_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$(TLIB$(2)_T_$(1)_H_$(3))/$(LIBRUSTC_GLOB_$(1)) $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTC_GLOB_$(4)),$$(notdir $$@))

$$(HLIB0_H_$(1))/$(CFG_RUSTLLVM_$(1)): \
		$$(TLIB$(2)_T_$(1)_H_$(3))/$(CFG_RUSTLLVM_$(1)) \
		| $(HLIB0_H_$(1))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

# Use stage1 to build other architectures: then you don't have to wait
# for stage2, but you get the latest updates to the compiler source.
$(foreach t,$(NON_BUILD_HOST),								\
 $(eval $(call BOOTSTRAP_STAGE0,$(t),1,$(CFG_BUILD))))
