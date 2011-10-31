# HOST_STAGE_N template: arg 1 is the N we're promoting *from*, arg 2
# is N+1. Must be invoked to promote target artifacts to host artifacts
# for stage 1-3 (stage0 host artifacts come from the snapshot).
#
# The easiest way to read this template is to assume we're promoting
# stage1 to stage2 and mentally gloss $(1) as 1, $(2) as 2.

define HOST_STAGE_N

# Host libraries and executables (stage$(2)/bin/rustc and its runtime needs)

$$(HOST_BIN$(2))/rustc$$(X): \
	$$(TARGET_HOST_BIN$(1))/rustc$$(X) \
	$$(HOST_LIB$(2))/$$(CFG_RUNTIME) \
	$$(HOST_LIB$(2))/$$(CFG_RUSTLLVM) \
	$$(HOST_STDLIB_DEFAULT$(2))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

# FIXME: The fuzzer depends on this. Remove once it's rpathed to correctly
# find it in the appropriate target directory
$$(HOST_LIB$(2))/$$(CFG_LIBRUSTC): \
	$$(TARGET_HOST_LIB$(1))/$$(CFG_LIBRUSTC) \
	$$(HOST_LIB$(2))/$$(CFG_RUNTIME) \
	$$(HOST_LIB$(2))/$$(CFG_RUSTLLVM) \
	$$(HOST_STDLIB_DEFAULT$(2))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HOST_LIB$(2))/$$(CFG_RUNTIME): \
	$$(TARGET_HOST_LIB$(1))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HOST_LIB$(2))/$$(CFG_STDLIB): \
	$$(TARGET_HOST_LIB$(1))/$$(CFG_STDLIB) \
	$$(HOST_LIB$(2))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

# FIXME: temporary
ifeq ($(2),0)
$$(HOST_LIB$(2))/$$(CFG_OLDSTDLIB): \
	$$(HOST_LIB$(2))/$$(CFG_STDLIB)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
endif

$$(HOST_LIB$(2))/libstd.rlib: \
	$$(TARGET_HOST_LIB$(1))/libstd.rlib \
	$$(HOST_LIB$(2))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HOST_LIB$(2))/$$(CFG_RUSTLLVM): \
	$$(TARGET_HOST_LIB$(1))/$$(CFG_RUSTLLVM)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

$(eval $(call HOST_STAGE_N,0,1))
$(eval $(call HOST_STAGE_N,1,2))
$(eval $(call HOST_STAGE_N,2,3))
