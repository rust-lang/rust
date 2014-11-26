# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

######################################################################
# Copy debugger related scripts
######################################################################

DEBUGGER_RUSTLIB_ETC_SCRIPTS=lldb_rust_formatters.py
DEBUGGER_BIN_SCRIPTS=rust-lldb

DEBUGGER_RUSTLIB_ETC_SCRIPTS_ABS=$(foreach script,$(DEBUGGER_RUSTLIB_ETC_SCRIPTS), \
                                     $(CFG_SRC_DIR)src/etc/$(script))
DEBUGGER_BIN_SCRIPTS_ABS=$(foreach script,$(DEBUGGER_BIN_SCRIPTS), \
                             $(CFG_SRC_DIR)src/etc/$(script))

DEBUGGER_SCRIPTS_ALL=$(DEBUGGER_RUSTLIB_ETC_SCRIPTS_ABS) $(DEBUGGER_BIN_SCRIPTS_ABS)

# $(1) - the stage to copy to
# $(2) - the host triple
define DEF_INSTALL_DEBUGGER_SCRIPTS_HOST

tmp/install-debugger-scripts$(1)_H_$(2).done: $$(DEBUGGER_SCRIPTS_ALL)
	$(Q)mkdir -p $$(HBIN$(1)_H_$(2))
	$(Q)mkdir -p $$(HLIB$(1)_H_$(2))/rustlib/etc
	$(Q)install $(DEBUGGER_BIN_SCRIPTS_ABS) $$(HBIN$(1)_H_$(2))
	$(Q)install $(DEBUGGER_RUSTLIB_ETC_SCRIPTS_ABS) $$(HLIB$(1)_H_$(2))/rustlib/etc
	$(Q)touch $$@
endef

# Expand host make-targets for all stages
$(foreach stage,$(STAGES), \
  $(foreach host,$(CFG_HOST), \
    $(eval $(call DEF_INSTALL_DEBUGGER_SCRIPTS_HOST,$(stage),$(host)))))

# $(1) is the stage number
# $(2) is the target triple
# $(3) is the host triple
define DEF_INSTALL_DEBUGGER_SCRIPTS_TARGET

tmp/install-debugger-scripts$(1)_T_$(2)_H_$(3).done: $$(DEBUGGER_SCRIPTS_ALL)
	$(Q)mkdir -p $$(TBIN$(1)_T_$(2)_H_$(3))
	$(Q)mkdir -p $$(TLIB$(1)_T_$(2)_H_$(3))/rustlib/etc
	$(Q)install $(DEBUGGER_BIN_SCRIPTS_ABS) $$(TBIN$(1)_T_$(2)_H_$(3))
	$(Q)install $(DEBUGGER_RUSTLIB_ETC_SCRIPTS_ABS) $$(TLIB$(1)_T_$(2)_H_$(3))/rustlib/etc
	$(Q)touch $$@
endef

# Expand target make-targets for all stages
$(foreach stage,$(STAGES), \
  $(foreach target,$(CFG_TARGET), \
    $(foreach host,$(CFG_HOST), \
      $(eval $(call DEF_INSTALL_DEBUGGER_SCRIPTS_TARGET,$(stage),$(target),$(host))))))
