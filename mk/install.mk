# FIXME: Docs are currently not installed from the stageN dirs.
# For consistency it might be desirable for stageN to be an exact
# mirror of the installation directory structure.

# Installation macro. Call with source directory as arg 1,
# destination directory as arg 2, and filename as arg 3
ifdef VERBOSE
 INSTALL = cp $(1)/$(3) $(2)/$(3)
else
 INSTALL = @$(call E, install: $(2)/$(3)) && cp $(1)/$(3) $(2)/$(3)
endif

# The stage we install from
ISTAGE = 3

PREFIX_ROOT = $(CFG_PREFIX)
PREFIX_BIN = $(PREFIX_ROOT)/bin
PREFIX_LIB = $(PREFIX_ROOT)/lib

define INSTALL_TARGET_N
  # $(1) is the target triple
  # $(2) is the host triple

# T{B,L} == Target {Bin, Lib} for stage ${ISTAGE}
TB$(1)_H_$(2) = $$(TBIN$$(ISTAGE)_T_$(1)_H_$(2))
TL$(1)_H_$(2) = $$(TLIB$$(ISTAGE))_T_$(1)_H_$(2))

# PT{R,B,L} == Prefix Target {Root, Bin, Lib}
PTR_T_$(1)_H_$(2) = $$(PREFIX_LIB)/rustc/$(1)
PTB_T_$(1)_H_$(2) = $$(PTR_T_$(1)_H_$(2))/bin
PTL_T_$(1)_H_$(2) = $$(PTR_T_$(1)_H_$(2))/lib

install-target-$(1)-host-$(2): $$(SREQ$$(ISTAGE)_T_$(1)_H_$(2))
	$(Q)mkdir -p $$(PTL_$(1)_H_$(2))
	$(Q)$(call INSTALL,$$(TL$(1)_H_$(2)),$$(PTL$(1)_H_$(2)),$$(CFG_RUNTIME))
	$(Q)$(call INSTALL,$$(TL$(1)_H_$(2)),$$(PTL$(1)_H_$(2)),$$(CFG_STDLIB))
	$(Q)$(call INSTALL,$$(TL$(1)_H_$(2)),$$(PTL$(1)_H_$(2)),intrinsics.bc)
	$(Q)$(call INSTALL,$$(TL$(1)_H_$(2)),$$(PTL$(1)_H_$(2)),libmorestack.a)
endef

$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call INSTALL_TARGET_N,$(target),$(CFG_HOST_TRIPLE))))

INSTALL_TARGET_RULES = $(foreach target,$(CFG_TARGET_TRIPLES), \
 install-target-$(target)-host-$(CFG_HOST_TRIPLE))

install: install-host install-targets

# Shorthand for build/stageN/bin
HB = $(HBIN$(ISTAGE)_H_$(HT))
# Shorthand for build/stageN/lib
HL = $(HLIB$(ISTAGE)_H_$(HT))
# Shorthand for the prefix bin directory
PHB = $(PREFIX_BIN)
# Shorthand for the prefix bin directory
PHL = $(PREFIX_LIB)

install-host: $(SREQ$(ISTAGE)$(CFG_HOST_TRIPLE))
	$(Q)mkdir -p $(PREFIX_BIN)
	$(Q)mkdir -p $(PREFIX_LIB)
	$(Q)mkdir -p $(PREFIX_ROOT)/share/man/man1
	$(Q)$(call INSTALL,$(HB),$(PHB),rustc$(X))
	$(Q)$(call INSTALL,$(HL),$(PHL),$(CFG_RUNTIME))
	$(Q)$(call INSTALL,$(HL),$(PHL),$(CFG_STDLIB))
	$(Q)$(call INSTALL,$(HL),$(PHL),$(CFG_RUSTLLVM))
	$(Q)$(call INSTALL,$(S)/man, \
	     $(PREFIX_ROOT)/share/man/man1,rustc.1)

install-targets: $(INSTALL_TARGET_RULES)
