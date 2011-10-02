# TARGET_STAGE_N template: This defines how target artifacts are built
# for all stage/target architecture combinations. Argument 1 is the
# stage and arg 2 is the target triple

# FIXME: We don't actually know how to build many of these when host
# and target architectures are not the same

define TARGET_STAGE_N

$$(TARGET_LIB$(1)$(2))/intrinsics.bc: $$(INTRINSICS_BC)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TARGET_LIB$(1)$(2))/main.o: rt/main.o
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TARGET_LIB$(1)$(2))/$$(CFG_STDLIB): \
	$$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
        $$(TARGET_SREQ$(1)$(2))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1))  --lib -o $$@ $$<

$$(TARGET_LIB$(1)$(2))/libstd.rlib: \
	$$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
        $$(TARGET_SREQ$(1)$(2))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)) --lib --static -o $$@ $$<

$$(TARGET_LIB$(1)$(2))/$$(CFG_RUNTIME): rt/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TARGET_LIB$(1)$(2))/$$(CFG_RUSTLLVM): rustllvm/$$(CFG_RUSTLLVM)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TARGET_BIN$(1)$(2))/rustc$$(X): \
	$$(COMPILER_CRATE) $$(COMPILER_INPUTS) \
	$$(TARGET_SREQ$(1)$(2)) \
	$$(TARGET_LIB$(1)$(2))/$$(CFG_RUSTLLVM) \
	$$(TARGET_STDLIB_DEFAULT$(1)$(2))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)) -o $$@ $$<

$$(TARGET_LIB$(1)$(2))/$$(CFG_LIBRUSTC): \
	$$(COMPILER_CRATE) $$(COMPILER_INPUTS) \
	$$(TARGET_SREQ$(1)$(2)) \
	$$(TARGET_LIB$(1)$(2))/$$(CFG_RUSTLLVM) \
	$$(TARGET_STDLIB_DEFAULT$(1)$(2))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)) --lib -o $$@ $$<

endef

# Instantiate template for all stages
$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call TARGET_STAGE_N,0,$(target))) \
 $(eval $(call TARGET_STAGE_N,1,$(target))) \
 $(eval $(call TARGET_STAGE_N,2,$(target))) \
 $(eval $(call TARGET_STAGE_N,3,$(target))))
