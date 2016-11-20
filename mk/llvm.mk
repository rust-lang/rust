# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.


ifdef CFG_ENABLE_FAST_MAKE
LLVM_DEPS := $(S)/.gitmodules
else

# This is just a rough approximation of LLVM deps
LLVM_DEPS_SRC=$(call rwildcard,$(CFG_LLVM_SRC_DIR)lib,*cpp *hpp)
LLVM_DEPS_INC=$(call rwildcard,$(CFG_LLVM_SRC_DIR)include,*cpp *hpp)
LLVM_DEPS=$(LLVM_DEPS_SRC) $(LLVM_DEPS_INC)
endif

ifdef CFG_DISABLE_OPTIMIZE_LLVM
LLVM_BUILD_CONFIG_MODE := Debug
else ifdef CFG_ENABLE_LLVM_RELEASE_DEBUGINFO
LLVM_BUILD_CONFIG_MODE := RelWithDebInfo
else
LLVM_BUILD_CONFIG_MODE := Release
endif

define DEF_LLVM_RULES

ifeq ($(1),$$(CFG_BUILD))
LLVM_DEPS_TARGET_$(1) := $$(LLVM_DEPS)
else
LLVM_DEPS_TARGET_$(1) := $$(LLVM_DEPS) $$(LLVM_CONFIG_$$(CFG_BUILD))
endif

# If CFG_LLVM_ROOT is defined then we don't build LLVM ourselves
ifeq ($(CFG_LLVM_ROOT),)

LLVM_STAMP_$(1) = $(S)src/rustllvm/llvm-auto-clean-trigger
LLVM_DONE_$(1) = $$(CFG_LLVM_BUILD_DIR_$(1))/llvm-finished-building

$$(LLVM_CONFIG_$(1)): $$(LLVM_DONE_$(1))

ifneq ($$(CFG_NINJA),)
BUILD_LLVM_$(1) := $$(CFG_NINJA) -C $$(CFG_LLVM_BUILD_DIR_$(1))
else ifeq ($$(findstring msvc,$(1)),msvc)
BUILD_LLVM_$(1) := $$(CFG_CMAKE) --build $$(CFG_LLVM_BUILD_DIR_$(1)) \
			--config $$(LLVM_BUILD_CONFIG_MODE)
else
BUILD_LLVM_$(1) := $$(MAKE) -C $$(CFG_LLVM_BUILD_DIR_$(1))
endif

$$(LLVM_DONE_$(1)): $$(LLVM_DEPS_TARGET_$(1)) $$(LLVM_STAMP_$(1))
	@$$(call E, cmake: llvm)
	$$(Q)if ! cmp $$(LLVM_STAMP_$(1)) $$(LLVM_DONE_$(1)); then \
		$$(MAKE) clean-llvm$(1); \
		$$(BUILD_LLVM_$(1)); \
	fi
	$$(Q)cp $$(LLVM_STAMP_$(1)) $$@

ifneq ($$(CFG_NINJA),)
clean-llvm$(1):
	@$$(call E, clean: llvm)
	$$(Q)$$(CFG_NINJA) -C $$(CFG_LLVM_BUILD_DIR_$(1)) -t clean
else ifeq ($$(findstring msvc,$(1)),msvc)
clean-llvm$(1):
	@$$(call E, clean: llvm)
	$$(Q)$$(CFG_CMAKE) --build $$(CFG_LLVM_BUILD_DIR_$(1)) \
		--config $$(LLVM_BUILD_CONFIG_MODE) \
		--target clean
else
clean-llvm$(1):
	@$$(call E, clean: llvm)
	$$(Q)$$(MAKE) -C $$(CFG_LLVM_BUILD_DIR_$(1)) clean
endif

else
clean-llvm$(1):
endif

$$(LLVM_AR_$(1)): $$(LLVM_CONFIG_$(1))

ifeq ($$(CFG_ENABLE_LLVM_STATIC_STDCPP),1)
LLVM_STDCPP_RUSTFLAGS_$(1) = -L "$$(dir $$(shell $$(CC_$(1)) $$(CFG_GCCISH_CFLAGS_$(1)) \
					-print-file-name=lib$(CFG_STDCPP_NAME).a))"
else
LLVM_STDCPP_RUSTFLAGS_$(1) =
endif


# LLVM linkage:
# Note: Filter with llvm-config so that optional targets which aren't present
# don't cause errors (ie PNaCl's target is only present within PNaCl's LLVM
# fork).
LLVM_LINKAGE_PATH_$(1):=$$(abspath $$(RT_OUTPUT_DIR_$(1))/llvmdeps.rs)
$$(LLVM_LINKAGE_PATH_$(1)): $(S)src/etc/mklldeps.py $$(LLVM_CONFIG_$(1))
	$(Q)$(CFG_PYTHON) "$$<" "$$@" "$$(filter $$(shell \
				$$(LLVM_CONFIG_$(1)) --components), \
                        $(LLVM_OPTIONAL_COMPONENTS)) $(LLVM_REQUIRED_COMPONENTS)" \
		"$$(CFG_ENABLE_LLVM_STATIC_STDCPP)" $$(LLVM_CONFIG_$(1)) \
		"$(CFG_STDCPP_NAME)" "$$(CFG_USING_LIBCPP)"
endef

$(foreach host,$(CFG_HOST), \
 $(eval $(call DEF_LLVM_RULES,$(host))))

$(foreach host,$(CFG_HOST), \
 $(eval LLVM_CONFIGS := $(LLVM_CONFIGS) $(LLVM_CONFIG_$(host))))

# This can't be done in target.mk because it's included before this file.
define LLVM_LINKAGE_DEPS
$$(TLIB$(1)_T_$(2)_H_$(3))/stamp.rustc_llvm: $$(LLVM_LINKAGE_PATH_$(2))
RUSTFLAGS$(1)_rustc_llvm_T_$(2) += $$(shell echo $$(LLVM_ALL_COMPONENTS_$(2)) | tr '-' '_' |\
	sed -e 's/^ //;s/\([^ ]*\)/\-\-cfg "llvm_component=\\"\1\\""/g')
endef

$(foreach source,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(eval $(call LLVM_LINKAGE_DEPS,0,$(target),$(source))) \
  $(eval $(call LLVM_LINKAGE_DEPS,1,$(target),$(source))) \
  $(eval $(call LLVM_LINKAGE_DEPS,2,$(target),$(source))) \
  $(eval $(call LLVM_LINKAGE_DEPS,3,$(target),$(source)))))
