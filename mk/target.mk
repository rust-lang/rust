# TARGET_STAGE_N template: This defines how target artifacts are built
# for all stage/target architecture combinations. The arguments:
# $(1) is the stage
# $(2) is the target triple
# $(3) is the host triple

# If you are making non-backwards compatible changes to the runtime
# (resp.  corelib), set this flag to 1.  It will cause stage1 to use
# the snapshot runtime (resp. corelib) rather than the runtime
# (resp. corelib) from the working directory.
USE_SNAPSHOT_RUNTIME=0
USE_SNAPSHOT_CORELIB=0
USE_SNAPSHOT_STDLIB=0

define TARGET_STAGE_N

$$(TLIB$(1)_T_$(2)_H_$(3))/libmorestack.a: \
		rt/$(2)/arch/$$(HOST_$(2))/libmorestack.a
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUSTLLVM): \
		rustllvm/$(2)/$$(CFG_RUSTLLVM)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(TBIN$(1)_T_$(2)_H_$(3))/rustc$$(X):				\
		$$(RUSTC_INPUTS)                                \
		$$(TLIBRUSTC_DEFAULT$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$<
ifdef CFG_ENABLE_PAX_FLAGS
	@$$(call E, apply PaX flags: $$@)
	@"$(CFG_PAXCTL)" -cm "$$@"
endif

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBRUSTC):		\
		$$(COMPILER_CRATE) $$(COMPILER_INPUTS)		\
                $$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBSYNTAX)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< && touch $$@

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBSYNTAX): \
                $$(LIBSYNTAX_CRATE) $$(LIBSYNTAX_INPUTS) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))			\
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUSTLLVM)	\
		$$(TCORELIB_DEFAULT$(1)_T_$(2)_H_$(3))      \
		$$(TSTDLIB_DEFAULT$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) $(BORROWCK) -o $$@ $$< && touch $$@

endef

# The stage0 (snapshot) compiler produces binaries that expect the
# snapshot runtime.  Normally the working directory runtime and
# snapshot runtime are compatible, so this is no problem. But
# sometimes we want to make non-backwards-compatible changes.  In
# those cases, the stage1 compiler and libraries (which are produced
# by stage0) should use the runtime from the snapshot.  The stage2
# compiler and libraries (which are produced by stage1) will be the
# first that are expecting to run against the runtime as defined in
# the working directory.
#
# The catch is that you may not add new functions to the runtime
# in this case!
#
# Arguments are the same as for TARGET_BASE_STAGE_N
define TARGET_RT_FROM_SNAPSHOT

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUNTIME): \
		$$(HLIB$(1)_H_$(3))/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

# This rule copies from the runtime for the working directory.  It
# applies to targets produced by stage1 or later.  See comment on
# previous rule.
#
# Arguments are the same as for TARGET_BASE_STAGE_N
define TARGET_RT_FROM_WD

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUNTIME): \
		rt/$(2)/$$(CFG_RUNTIME)
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

# As above, but builds the corelib either by taking it out of the
# snapshot or from the working directory.

define TARGET_CORELIB_FROM_SNAPSHOT

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_CORELIB): \
		$$(HLIB$(1)_H_$(3))/$$(CFG_CORELIB) \
		$$(CORELIB_INPUTS) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
	$$(Q)cp $$(HLIB$(1)_H_$(3))/$$(CORELIB_GLOB) \
		$$(TLIB$(1)_T_$(2)_H_$(3))

endef

define TARGET_CORELIB_FROM_WD

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_CORELIB): \
		$$(CORELIB_CRATE) $$(CORELIB_INPUTS) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< && touch $$@

endef

define TARGET_STDLIB_FROM_SNAPSHOT

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_STDLIB): \
		$$(HLIB$(1)_H_$(3))/$$(CFG_STDLIB) \
		$$(STDLIB_INPUTS) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@
	$$(Q)cp $$(HLIB$(1)_H_$(3))/$$(STDLIB_GLOB) \
		$$(TLIB$(1)_T_$(2)_H_$(3))

endef

define TARGET_STDLIB_FROM_WD

$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_STDLIB): \
		$$(STDLIB_CRATE) $$(STDLIB_INPUTS) \
	        $$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_CORELIB) \
		$$(TSREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< && touch $$@

endef

# In principle, each host can build each target:
$(foreach source,$(CFG_TARGET_TRIPLES),				\
 $(foreach target,$(CFG_TARGET_TRIPLES),			\
  $(eval $(call TARGET_STAGE_N,0,$(target),$(source)))		\
  $(eval $(call TARGET_STAGE_N,1,$(target),$(source)))		\
  $(eval $(call TARGET_STAGE_N,2,$(target),$(source)))		\
  $(eval $(call TARGET_STAGE_N,3,$(target),$(source)))))

# Host triple either uses the snapshot runtime or runtime from
# working directory, depending on the USE_SNAPSHOT_RUNTIME var.
ifeq ($(USE_SNAPSHOT_RUNTIME),1)
    $(foreach src,$(CFG_HOST_TRIPLE),\
		$(eval $(call TARGET_RT_FROM_SNAPSHOT,0,$(src),$(src))))
else 
    $(foreach src,$(CFG_HOST_TRIPLE),\
		$(eval $(call TARGET_RT_FROM_WD,0,$(src),$(src))))
endif

ifeq ($(USE_SNAPSHOT_CORELIB),1)
    $(foreach src,$(CFG_HOST_TRIPLE),\
		$(eval $(call TARGET_CORELIB_FROM_SNAPSHOT,0,$(src),$(src))))
else 
    $(foreach src,$(CFG_HOST_TRIPLE),\
		$(eval $(call TARGET_CORELIB_FROM_WD,0,$(src),$(src))))
endif

ifeq ($(USE_SNAPSHOT_STDLIB),1)
    $(foreach src,$(CFG_HOST_TRIPLE),\
		$(eval $(call TARGET_STDLIB_FROM_SNAPSHOT,0,$(src),$(src))))
else
    $(foreach src,$(CFG_HOST_TRIPLE),\
		$(eval $(call TARGET_STDLIB_FROM_WD,0,$(src),$(src))))
endif

# Non-host triples build the stage0 runtime from the working directory
$(foreach source,$(CFG_TARGET_TRIPLES),				\
 $(foreach target,$(NON_HOST_TRIPLES),				\
  $(eval $(call TARGET_RT_FROM_WD,0,$(target),$(source)))       \
  $(eval $(call TARGET_CORELIB_FROM_WD,0,$(target),$(source)))  \
  $(eval $(call TARGET_STDLIB_FROM_WD,0,$(target),$(source)))  \
))

# After stage0, always build the stage0 runtime from the working directory
$(foreach source,$(CFG_TARGET_TRIPLES),				\
 $(foreach target,$(CFG_TARGET_TRIPLES),			\
  $(eval $(call TARGET_RT_FROM_WD,1,$(target),$(source)))	\
  $(eval $(call TARGET_RT_FROM_WD,2,$(target),$(source)))	\
  $(eval $(call TARGET_RT_FROM_WD,3,$(target),$(source)))  	\
  $(eval $(call TARGET_CORELIB_FROM_WD,1,$(target),$(source)))	\
  $(eval $(call TARGET_CORELIB_FROM_WD,2,$(target),$(source)))	\
  $(eval $(call TARGET_CORELIB_FROM_WD,3,$(target),$(source)))	\
  $(eval $(call TARGET_STDLIB_FROM_WD,1,$(target),$(source)))	\
  $(eval $(call TARGET_STDLIB_FROM_WD,2,$(target),$(source)))	\
  $(eval $(call TARGET_STDLIB_FROM_WD,3,$(target),$(source)))	\
))

