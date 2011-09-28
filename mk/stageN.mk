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

STAGE$(2) := $$(Q)$$(call CFG_RUN_TARG,stage$(2), \
                $$(CFG_VALGRIND_COMPILE) stage$(2)/rustc$$(X) \
                $$(CFG_RUSTC_FLAGS))

PERF_STAGE$(2) := $$(Q)$$(call CFG_RUN_TARG,stage$(2), \
                $$(CFG_PERF_TOOL) stage$(2)/rustc$$(X) \
                $$(CFG_RUSTC_FLAGS))

stage$(2)/%.o: stage$(2)/%.s
	@$$(call E, assemble [gcc]: $$@)
	$$(Q)gcc $$(CFG_GCCISH_CFLAGS) -o $$@ -c $$<

stage$(2)/rustc$$(X): $$(COMPILER_CRATE) $$(COMPILER_INPUTS)          \
                      stage$(2)/$$(CFG_RUNTIME)                       \
                      $$(call CFG_STDLIB_DEFAULT,stage$(1),stage$(2)) \
                      stage$(2)/$$(CFG_RUSTLLVM)                      \
                      $$(SREQ$(1))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)) -L stage$(2) -o $$@ $$<

stage$(2)/$$(CFG_RUNTIME): rt/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

stage$(2)/$$(CFG_STDLIB): stage$(1)/lib/$$(CFG_STDLIB)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

stage$(2)/$$(CFG_RUSTLLVM): rustllvm/$$(CFG_RUSTLLVM)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

# Expand out target libraries

$(eval $(call TARGET_LIBS,$(1),$(2),$(3)))

endef


define TARGET_LIBS
stage$(2)/lib/intrinsics.bc: $$(INTRINSICS_BC)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

stage$(2)/lib/main.o: rt/main.o
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

stage$(2)/lib/$$(CFG_LIBRUSTC): $$(COMPILER_CRATE) $$(COMPILER_INPUTS) \
                                $$(SREQ$(2))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2)) --lib -o $$@ $$<

stage$(2)/lib/$$(CFG_STDLIB): $$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
                              stage$(2)/rustc$$(X)               \
                              stage$(2)/$$(CFG_RUNTIME)          \
                              stage$(2)/$$(CFG_RUSTLLVM)         \
                              stage$(2)/lib/intrinsics.bc        \
                              $$(SREQ$(1))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2))  --lib -o $$@ $$<

stage$(2)/lib/libstd.rlib: $$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
                           stage$(2)/rustc$$(X)               \
                           stage$(2)/$$(CFG_RUNTIME)          \
                           stage$(2)/$$(CFG_RUSTLLVM)         \
                           stage$(2)/lib/intrinsics.bc        \
                           $$(SREQ$(1))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2)) --lib --static -o $$@ $$<

stage$(2)/lib/$$(CFG_RUNTIME): rt/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@


# New per-target-arch target libraries; when we've transitioned to
# using these exclusively, you should delete the non-arch-prefixed
# rules above. They're duplicates, redundant.

stage$(2)/lib/rustc/$(3)/intrinsics.bc: $$(INTRINSICS_BC)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

stage$(2)/lib/rustc/$(3)/main.o: rt/main.o
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

stage$(2)/lib/rustc/$(3)/$$(CFG_LIBRUSTC): \
          $$(COMPILER_CRATE) $$(COMPILER_INPUTS) \
          $$(SREQ$(2))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2)) --lib -o $$@ $$<

stage$(2)/lib/rustc/$(3)/$$(CFG_STDLIB): \
        $$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
        stage$(2)/rustc$$(X)               \
        stage$(2)/$$(CFG_RUNTIME)          \
        stage$(2)/$$(CFG_RUSTLLVM)         \
        stage$(2)/lib/intrinsics.bc        \
        $$(SREQ$(1))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2))  --lib -o $$@ $$<

stage$(2)/lib/rustc/$(3)/libstd.rlib: \
        $$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
        stage$(2)/rustc$$(X)               \
        stage$(2)/$$(CFG_RUNTIME)          \
        stage$(2)/$$(CFG_RUSTLLVM)         \
        stage$(2)/lib/intrinsics.bc        \
        $$(SREQ$(1))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2)) --lib --static -o $$@ $$<

stage$(2)/lib/rustc/$(3)/$$(CFG_RUNTIME): rt/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef



# Instantiate template for 0->1, 1->2, 2->3 build dirs
$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call STAGE_N,0,1,$(target)))   \
 $(eval $(call STAGE_N,1,2,$(target)))   \
 $(eval $(call STAGE_N,2,3,$(target))))
