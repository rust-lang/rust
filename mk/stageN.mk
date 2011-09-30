# STAGE_N template: arg 1 is the N we're building *from*, arg 2 is N+1, arg 3
# is the target triple we're building for. You have to invoke this for each
# target triple.
#
# The easiest way to read this template is to assume we're building stage2
# using stage1, and mentally gloss $(1) as 1, $(2) as 2.
#
# TARGET_LIBS is pulled out seperately because we need to specially invoke
# it to build stage0/lib/libstd using stage0/rustc and to use the
# new rustrt in stage0/lib/.

define STAGE_N

# Host libraries and executables (stage$(2)/rustc and its runtime needs)
#
# NB: Due to make not wanting to run the same implicit rules twice on the same
# rule tree (implicit-rule recursion prevention, see "Chains of Implicit
# Rules" in GNU Make manual) we have to re-state the %.o and %.s patterns here
# for different directories, to handle cases where (say) a test relies on a
# compiler that relies on a .o file.

$$(HOST_BIN$(2))/%.o: $$(HOST_BIN$(2))/%.s
	@$$(call E, assemble [gcc]: $$@)
	$$(Q)gcc $$(CFG_GCCISH_CFLAGS) -o $$@ -c $$<

$$(HOST_LIB$(2))/%.o: $$(HOST_LIB$(2))/%.s
	@$$(call E, assemble [gcc]: $$@)
	$$(Q)gcc $$(CFG_GCCISH_CFLAGS) -o $$@ -c $$<

$$(HOST_BIN$(2))/rustc$$(X): $$(COMPILER_CRATE) $$(COMPILER_INPUTS)          \
                          $$(HOST_LIB$(2))/$$(CFG_RUNTIME)                       \
                          $$(HOST_STDLIB_DEFAULT$(2)) \
                          $$(HOST_LIB$(2))/$$(CFG_RUSTLLVM)                      \
                          $$(SREQ$(1)$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)) -L $$(HOST_LIB$(2)) -o $$@ $$<

$$(HOST_LIB$(2))/$$(CFG_LIBRUSTC): \
          $$(COMPILER_CRATE) $$(COMPILER_INPUTS) \
          $$(SREQ$(2)$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)) -L $$(HOST_LIB$(2)) --lib -o $$@ $$<

$$(HOST_LIB$(2))/$$(CFG_RUNTIME): rt/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HOST_LIB$(2))/$$(CFG_STDLIB): $$(TARGET_HOST_LIB$(1))/$$(CFG_STDLIB)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HOST_LIB$(2))/libstd.rlib: $$(TARGET_HOST_LIB$(1))/libstd.rlib
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HOST_LIB$(2))/$$(CFG_RUSTLLVM): rustllvm/$$(CFG_RUSTLLVM)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

# Expand out target libraries

$(eval $(call TARGET_LIBS,$(1),$(2),$(3)))

endef


define TARGET_LIBS

# New per-target-arch target libraries; when we've transitioned to
# using these exclusively, you should delete the non-arch-prefixed
# rules above. They're duplicates, redundant.

$$(TARGET_LIB$(2)$(3))/intrinsics.bc: $$(INTRINSICS_BC)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TARGET_LIB$(2)$(3))/main.o: rt/main.o
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TARGET_LIB$(2)$(3))/$$(CFG_STDLIB): \
        $$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
        $$(HOST_BIN$(2))/rustc$$(X)               \
        $$(HOST_LIB$(2))/$$(CFG_RUNTIME)          \
        $$(HOST_LIB$(2))/$$(CFG_RUSTLLVM)         \
        $$(TARGET_LIB$(2)$(3))/intrinsics.bc        \
        $$(SREQ$(1)$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2))  --lib -o $$@ $$<

$$(TARGET_LIB$(2)$(3))/libstd.rlib: \
        $$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
        $$(HOST_BIN$(2))/rustc$$(X)               \
        $$(HOST_LIB$(2))/$$(CFG_RUNTIME)          \
        $$(HOST_LIB$(2))/$$(CFG_RUSTLLVM)         \
        $$(TARGET_LIB$(2)$(3))/intrinsics.bc        \
        $$(SREQ$(1)$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2)) --lib --static -o $$@ $$<

$$(TARGET_LIB$(2)$(3))/$$(CFG_RUNTIME): rt/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef



# Instantiate template for 0->1, 1->2, 2->3 build dirs
$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call STAGE_N,0,1,$(target)))   \
 $(eval $(call STAGE_N,1,2,$(target)))   \
 $(eval $(call STAGE_N,2,3,$(target))))
