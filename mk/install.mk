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

# Installation macros.
# For INSTALL,
# $(1) is the source dirctory
# $(2) is the destination directory
# $(3) is the filename/libname-glob
ifdef VERBOSE
 INSTALL = install -m755 $(1)/$(3) $(DESTDIR)$(2)/$(3)
else
 INSTALL = $(Q)$(call E, install: $(DESTDIR)$(2)/$(3)) && install -m755 $(1)/$(3) $(DESTDIR)$(2)/$(3)
endif

# For MK_INSTALL_DIR
# $(1) is the directory to create
MK_INSTALL_DIR = (umask 022 && mkdir -p $(DESTDIR)$(1))

# For INSTALL_LIB,
# Target-specific $(LIB_SOURCE_DIR) is the source directory
# Target-specific $(LIB_DESTIN_DIR) is the destination directory
# $(1) is the filename/libname-glob
ifdef VERBOSE
 DO_INSTALL_LIB = install -m644 `ls -drt1 $(LIB_SOURCE_DIR)/$(1) | tail -1` $(DESTDIR)$(LIB_DESTIN_DIR)/
else
 DO_INSTALL_LIB = $(Q)$(call E, install_lib: $(DESTDIR)$(LIB_DESTIN_DIR)/$(1)) &&                    \
	       install -m644 `ls -drt1 $(LIB_SOURCE_DIR)/$(1) | tail -1` $(DESTDIR)$(LIB_DESTIN_DIR)/
endif

# Target-specific $(LIB_SOURCE_DIR) is the source directory
# Target-specific $(LIB_DESTIN_DIR) is the destination directory
# $(1) is the filename/libname-glob
define INSTALL_LIB
  $(if $(filter-out 1,$(words $(wildcard $(LIB_SOURCE_DIR)/$(1)))),        \
       $(error Aborting install because more than one library matching     \
               $(1) is present in build tree $(LIB_SOURCE_DIR):            \
               $(wildcard $(LIB_SOURCE_DIR)/$(1))))
  $(Q)LIB_NAME="$(notdir $(lastword $(wildcard $(LIB_SOURCE_DIR)/$(1))))"; \
  MATCHES="$(filter-out %$(notdir $(lastword $(wildcard $(LIB_SOURCE_DIR)/$(1)))),\
                        $(wildcard $(LIB_DESTIN_DIR)/$(1)))";              \
  if [ -n "$$MATCHES" ]; then                                              \
    echo "warning: one or libraries matching Rust library '$(1)'" &&       \
    echo "  (other than '$$LIB_NAME' itself) already present"     &&       \
    echo "  at destination $(LIB_DESTIN_DIR):"                    &&       \
    echo $$MATCHES ;                                                       \
  fi
  $(call DO_INSTALL_LIB,$(1))
endef

# The stage we install from
ISTAGE = 2

PREFIX_ROOT = $(CFG_PREFIX)
PREFIX_BIN = $(PREFIX_ROOT)/bin
PREFIX_LIB = $(PREFIX_ROOT)/$(CFG_LIBDIR)

define INSTALL_PREPARE_N
  # $(1) is the target triple
  # $(2) is the host triple

# T{B,L} == Target {Bin, Lib} for stage ${ISTAGE}
TB$(1)$(2) = $$(TBIN$$(ISTAGE)_T_$(1)_H_$(2))
TL$(1)$(2) = $$(TLIB$$(ISTAGE)_T_$(1)_H_$(2))

# PT{R,B,L} == Prefix Target {Root, Bin, Lib}
PTR$(1)$(2) = $$(PREFIX_LIB)/$(CFG_RUSTLIBDIR)/$(1)
PTB$(1)$(2) = $$(PTR$(1)$(2))/bin
PTL$(1)$(2) = $$(PTR$(1)$(2))/$(CFG_LIBDIR)

endef

$(foreach target,$(CFG_TARGET), \
 $(eval $(call INSTALL_PREPARE_N,$(target),$(CFG_BUILD))))

define INSTALL_TARGET_N
install-target-$(1)-host-$(2): LIB_SOURCE_DIR=$$(TL$(1)$(2))
install-target-$(1)-host-$(2): LIB_DESTIN_DIR=$$(PTL$(1)$(2))
install-target-$(1)-host-$(2): $$(TSREQ$$(ISTAGE)_T_$(1)_H_$(2)) $$(SREQ$$(ISTAGE)_T_$(1)_H_$(2))
	$$(Q)$$(call MK_INSTALL_DIR,$$(PTL$(1)$(2)))
	$$(Q)$$(call INSTALL_LIB,$$(STDLIB_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(STDLIB_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(EXTRALIB_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(EXTRALIB_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBRUSTUV_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBRUSTUV_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBGREEN_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBGREEN_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBNATIVE_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBNATIVE_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,libmorestack.a)

endef

define INSTALL_HOST_N
install-target-$(1)-host-$(2): LIB_SOURCE_DIR=$$(TL$(1)$(2))
install-target-$(1)-host-$(2): LIB_DESTIN_DIR=$$(PTL$(1)$(2))
install-target-$(1)-host-$(2): $$(CSREQ$$(ISTAGE)_T_$(1)_H_$(2))
	$$(Q)$$(call MK_INSTALL_DIR,$$(PTL$(1)$(2)))
	$$(Q)$$(call INSTALL_LIB,$$(STDLIB_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(STDLIB_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(EXTRALIB_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(EXTRALIB_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBRUSTUV_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBRUSTUV_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBGREEN_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBGREEN_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBNATIVE_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBNATIVE_RGLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBRUSTC_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBSYNTAX_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBRUSTPKG_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,$$(LIBRUSTDOC_GLOB_$(1)))
	$$(Q)$$(call INSTALL_LIB,libmorestack.a)

endef

$(foreach target,$(CFG_TARGET), \
 $(if $(findstring $(target), $(CFG_BUILD)), \
  $(eval $(call INSTALL_HOST_N,$(target),$(CFG_BUILD))), \
  $(eval $(call INSTALL_TARGET_N,$(target),$(CFG_BUILD)))))

INSTALL_TARGET_RULES = $(foreach target,$(CFG_TARGET), \
 install-target-$(target)-host-$(CFG_BUILD))

install: all install-host install-targets

# Shorthand for build/stageN/bin
HB = $(HBIN$(ISTAGE)_H_$(CFG_BUILD))
HB2 = $(HBIN2_H_$(CFG_BUILD))
# Shorthand for build/stageN/lib
HL = $(HLIB$(ISTAGE)_H_$(CFG_BUILD))
# Shorthand for the prefix bin directory
PHB = $(PREFIX_BIN)
# Shorthand for the prefix bin directory
PHL = $(PREFIX_LIB)

install-host: LIB_SOURCE_DIR=$(HL)
install-host: LIB_DESTIN_DIR=$(PHL)
install-host: $(CSREQ$(ISTAGE)_T_$(CFG_BUILD_)_H_$(CFG_BUILD_))
	$(Q)$(call MK_INSTALL_DIR,$(PREFIX_BIN))
	$(Q)$(call MK_INSTALL_DIR,$(PREFIX_LIB))
	$(Q)$(call MK_INSTALL_DIR,$(CFG_MANDIR)/man1)
	$(Q)$(call INSTALL,$(HB2),$(PHB),rustc$(X_$(CFG_BUILD)))
	$(Q)$(call INSTALL,$(HB2),$(PHB),rustpkg$(X_$(CFG_BUILD)))
	$(Q)$(call INSTALL,$(HB2),$(PHB),rustdoc$(X_$(CFG_BUILD)))
	$(Q)$(call INSTALL_LIB,$(STDLIB_GLOB_$(CFG_BUILD)))
	$(Q)$(call INSTALL_LIB,$(EXTRALIB_GLOB_$(CFG_BUILD)))
	$(Q)$(call INSTALL_LIB,$(LIBRUSTUV_GLOB_$(CFG_BUILD)))
	$(Q)$(call INSTALL_LIB,$(LIBGREEN_GLOB_$(CFG_BUILD)))
	$(Q)$(call INSTALL_LIB,$(LIBRUSTC_GLOB_$(CFG_BUILD)))
	$(Q)$(call INSTALL_LIB,$(LIBSYNTAX_GLOB_$(CFG_BUILD)))
	$(Q)$(call INSTALL_LIB,$(LIBRUSTPKG_GLOB_$(CFG_BUILD)))
	$(Q)$(call INSTALL_LIB,$(LIBRUSTDOC_GLOB_$(CFG_BUILD)))
	$(Q)$(call INSTALL,$(S)/man,$(CFG_MANDIR)/man1,rustc.1)
	$(Q)$(call INSTALL,$(S)/man,$(CFG_MANDIR)/man1,rustdoc.1)
	$(Q)$(call INSTALL,$(S)/man,$(CFG_MANDIR)/man1,rustpkg.1)

install-targets: $(INSTALL_TARGET_RULES)


HOST_LIB_FROM_HL_GLOB = \
  $(patsubst $(HL)/%,$(PHL)/%,$(wildcard $(HL)/$(1)))

uninstall:
	$(Q)rm -f $(PHB)/rustc$(X_$(CFG_BUILD))
	$(Q)rm -f $(PHB)/rustpkg$(X_$(CFG_BUILD))
	$(Q)rm -f $(PHB)/rustdoc$(X_$(CFG_BUILD))
	$(Q)for i in \
          $(call HOST_LIB_FROM_HL_GLOB,$(STDLIB_GLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(STDLIB_RGLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(EXTRALIB_GLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(EXTRALIB_RGLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBRUSTUV_GLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBRUSTUV_RGLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBGREEN_GLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBGREEN_RGLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBNATIVE_GLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBNATIVE_RGLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBRUSTC_GLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBSYNTAX_GLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBRUSTPKG_GLOB_$(CFG_BUILD))) \
          $(call HOST_LIB_FROM_HL_GLOB,$(LIBRUSTDOC_GLOB_$(CFG_BUILD))) \
        ; \
        do rm -f $$i ; \
        done
	$(Q)rm -Rf $(PHL)/rustc
	$(Q)rm -f $(CFG_MANDIR)/man1/rustc.1
	$(Q)rm -f $(CFG_MANDIR)/man1/rustdoc.1
	$(Q)rm -f $(CFG_MANDIR)/man1/rusti.1
	$(Q)rm -f $(CFG_MANDIR)/man1/rustpkg.1

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
	$(Q)$(call ADB_SHELL,mkdir,$(CFG_RUNTIME_PUSH_DIR))
	$(Q)$(call ADB_PUSH,$$(TL$(1)$(2))/$$(STDLIB_GLOB_$(1)),$(CFG_RUNTIME_PUSH_DIR))
	$(Q)$(call ADB_PUSH,$$(TL$(1)$(2))/$$(EXTRALIB_GLOB_$(1)),$(CFG_RUNTIME_PUSH_DIR))
	$(Q)$(call ADB_PUSH,$$(TL$(1)$(2))/$$(LIBRUSTUV_GLOB_$(1)),$(CFG_RUNTIME_PUSH_DIR))
	$(Q)$(call ADB_PUSH,$$(TL$(1)$(2))/$$(LIBGREEN_GLOB_$(1)),$(CFG_RUNTIME_PUSH_DIR))
endef

define INSTALL_RUNTIME_TARGET_CLEANUP_N
install-runtime-target-$(1)-cleanup:
	$(Q)$(call ADB,remount)
	$(Q)$(call ADB_SHELL,rm,$(CFG_RUNTIME_PUSH_DIR)/$(STDLIB_GLOB_$(1)))
	$(Q)$(call ADB_SHELL,rm,$(CFG_RUNTIME_PUSH_DIR)/$(EXTRALIB_GLOB_$(1)))
	$(Q)$(call ADB_SHELL,rm,$(CFG_RUNTIME_PUSH_DIR)/$(LIBRUSTUV_GLOB_$(1)))
	$(Q)$(call ADB_SHELL,rm,$(CFG_RUNTIME_PUSH_DIR)/$(LIBGREEN_GLOB_$(1)))
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
