# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# FIXME: Docs are currently not installed from the stageN dirs.
# For consistency it might be desirable for stageN to be an exact
# mirror of the installation directory structure.

# The stage we install from
ISTAGE = $(PREPARE_STAGE)

$(eval $(call DEF_PREPARE,mkfile-install))

install: PREPARE_HOST=$(CFG_BUILD)
install: PREPARE_TARGETS=$(CFG_TARGET)
install: PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
install: PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
install: PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
install: PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
install: PREPARE_SOURCE_DIR=$(PREPARE_HOST)/stage$(PREPARE_STAGE)
install: PREPARE_SOURCE_BIN_DIR=$(PREPARE_SOURCE_DIR)/bin
install: PREPARE_SOURCE_LIB_DIR=$(PREPARE_SOURCE_DIR)/$(CFG_LIBDIR_RELATIVE)
install: PREPARE_SOURCE_MAN_DIR=$(S)/man
install: PREPARE_DEST_BIN_DIR=$(DESTDIR)$(CFG_PREFIX)/bin
install: PREPARE_DEST_LIB_DIR=$(DESTDIR)$(CFG_LIBDIR)
install: PREPARE_DEST_MAN_DIR=$(DESTDIR)$(CFG_MANDIR)/man1
install: prepare-everything-mkfile-install


# Uninstall code

PREFIX_ROOT = $(CFG_PREFIX)
PREFIX_BIN = $(PREFIX_ROOT)/bin
PREFIX_LIB = $(CFG_LIBDIR)

INSTALL_TOOLS := $(PREPARE_TOOLS)

# Shorthand for build/stageN/bin
HB = $(HBIN$(ISTAGE)_H_$(CFG_BUILD))
HB2 = $(HBIN2_H_$(CFG_BUILD))
# Shorthand for build/stageN/lib
HL = $(HLIB$(ISTAGE)_H_$(CFG_BUILD))
# Shorthand for the prefix bin directory
PHB = $(PREFIX_BIN)
# Shorthand for the prefix bin directory
PHL = $(PREFIX_LIB)

HOST_LIB_FROM_HL_GLOB = \
  $(patsubst $(HL)/%,$(PHL)/%,$(wildcard $(HL)/$(1)))

uninstall: $(foreach tool,$(INSTALL_TOOLS),uninstall-tool-$(tool))
	$(Q)rm -Rf $(PHL)/$(CFG_RUSTLIBDIR)

define UNINSTALL_TOOL
uninstall-tool-$(1): $$(foreach dep,$$(TOOL_DEPS_$(1)),uninstall-lib-$$(dep))
	$$(Q)rm -f $$(PHB)/$(1)$$(X_$$(CFG_BUILD))
	$$(Q)rm -f $$(CFG_MANDIR)/man1/$(1).1
endef

$(foreach tool,$(INSTALL_TOOLS),$(eval $(call UNINSTALL_TOOL,$(tool))))

define UNINSTALL_LIB
uninstall-lib-$(1): $$(foreach dep,$$(RUST_DEPS_$(1)),uninstall-lib-$$(dep))
	$$(Q)rm -f $$(call HOST_LIB_FROM_HL_GLOB,$$(call CFG_LIB_GLOB_$$(CFG_BUILD),$(1)))
endef

$(foreach lib,$(CRATES),$(eval $(call UNINSTALL_LIB,$(lib))))


# Android runtime setup
# FIXME: This probably belongs somewhere else

# target platform specific variables
# for arm-linux-androidabi
define DEF_ADB_DEVICE_STATUS
CFG_ADB_DEVICE_STATUS=$(1)
endef

$(foreach target,$(CFG_TARGET), \
  $(if $(findstring $(target),"arm-linux-androideabi"), \
    $(if $(findstring adb,$(CFG_ADB)), \
      $(if $(findstring device,$(shell $(CFG_ADB) devices 2>/dev/null | grep -E '^[_A-Za-z0-9-]+[[:blank:]]+device')), \
        $(info install: install-runtime-target for $(target) enabled \
          $(info install: android device attached) \
          $(eval $(call DEF_ADB_DEVICE_STATUS, true))), \
        $(info install: install-runtime-target for $(target) disabled \
          $(info install: android device not attached) \
          $(eval $(call DEF_ADB_DEVICE_STATUS, false))) \
      ), \
      $(info install: install-runtime-target for $(target) disabled \
        $(info install: adb not found) \
        $(eval $(call DEF_ADB_DEVICE_STATUS, false))) \
    ), \
  ) \
)

ifeq (install-runtime-target,$(firstword $(MAKECMDGOALS)))
$(eval $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)):;@:)
L_TOKEN := $(word 2,$(MAKECMDGOALS))
ifeq ($(L_TOKEN),)
CFG_RUNTIME_PUSH_DIR=/system/lib
else
CFG_RUNTIME_PUSH_DIR=$(L_TOKEN)
endif

ifeq ($(CFG_ADB_DEVICE_STATUS),true)
ifdef VERBOSE
 ADB = adb $(1)
 ADB_PUSH = adb push $(1) $(2)
 ADB_SHELL = adb shell $(1) $(2)
else
 ADB = $(Q)$(call E, adb $(1)) && adb $(1) 1>/dev/null
 ADB_PUSH = $(Q)$(call E, adb push $(1)) && adb push $(1) $(2) 1>/dev/null
 ADB_SHELL = $(Q)$(call E, adb shell $(1) $(2)) && adb shell $(1) $(2) 1>/dev/null
endif

define INSTALL_RUNTIME_TARGET_N
install-runtime-target-$(1)-host-$(2): $$(TSREQ$$(ISTAGE)_T_$(1)_H_$(2)) $$(SREQ$$(ISTAGE)_T_$(1)_H_$(2))
	$$(Q)$$(call ADB_SHELL,mkdir,$(CFG_RUNTIME_PUSH_DIR))
	$$(Q)$$(foreach crate,$$(TARGET_CRATES),\
	    $$(call ADB_PUSH,$$(TL$(1)$(2))/$$(call CFG_LIB_GLOB_$(1),$$(crate)),\
			$$(CFG_RUNTIME_PUSH_DIR));)
endef

define INSTALL_RUNTIME_TARGET_CLEANUP_N
install-runtime-target-$(1)-cleanup:
	$$(Q)$$(call ADB,remount)
	$$(Q)$$(foreach crate,$$(TARGET_CRATES),\
	    $$(call ADB_SHELL,rm,$$(CFG_RUNTIME_PUSH_DIR)/$$(call CFG_LIB_GLOB_$(1),$$(crate)));)
endef

$(eval $(call INSTALL_RUNTIME_TARGET_N,arm-linux-androideabi,$(CFG_BUILD)))
$(eval $(call INSTALL_RUNTIME_TARGET_CLEANUP_N,arm-linux-androideabi))

install-runtime-target: \
	install-runtime-target-arm-linux-androideabi-cleanup \
	install-runtime-target-arm-linux-androideabi-host-$(CFG_BUILD)
else
install-runtime-target:
	@echo "No device to install runtime library"
	@echo
endif
endif
