# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

ifdef CFG_DISABLE_VERIFY_INSTALL
MAYBE_DISABLE_VERIFY=--disable-verify
else
MAYBE_DISABLE_VERIFY=
endif

install: dist-install-dir-$(CFG_BUILD)-with-target-libs | tmp/empty_dir
	$(Q)cd tmp/empty_dir && sh ../../tmp/dist/$(PKG_NAME)-$(CFG_BUILD)/install.sh --prefix="$(DESTDIR)$(CFG_PREFIX)" --libdir="$(DESTDIR)$(CFG_LIBDIR)" --mandir="$(DESTDIR)$(CFG_MANDIR)" "$(MAYBE_DISABLE_VERIFY)"
# Remove tmp files while we can because they may have been created under sudo
	$(Q)rm -R tmp/dist

uninstall: dist-install-dir-$(CFG_BUILD)-with-target-libs | tmp/empty_dir
	$(Q)cd tmp/empty_dir && sh ../../tmp/dist/$(PKG_NAME)-$(CFG_BUILD)/install.sh --uninstall --prefix="$(DESTDIR)$(CFG_PREFIX)" --libdir="$(DESTDIR)$(CFG_LIBDIR)" --mandir="$(DESTDIR)$(CFG_MANDIR)"
# Remove tmp files while we can because they may have been created under sudo
	$(Q)rm -R tmp/dist

tmp/empty_dir:
	mkdir -p $@

######################################################################
# Android remote installation
######################################################################

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
