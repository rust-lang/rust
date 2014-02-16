# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Basic support for producing installation images.
#
# The 'prepare' build target copies all release artifacts from the build
# directory to some other location, placing all binaries, libraries, and
# docs in their final locations relative to each other.
#
# It requires the following variables to be set:
#
#   PREPARE_HOST - the host triple 
#   PREPARE_TARGETS - the target triples, space separated
#   PREPARE_DEST_DIR - the directory to put the image

prepare: PREPARE_STAGE=2
prepare: PREPARE_DIR_CMD=$(DEFAULT_PREPARE_DIR_CMD)
prepare: PREPARE_BIN_CMD=$(DEFAULT_PREPARE_BIN_CMD)
prepare: PREPARE_LIB_CMD=$(DEFAULT_PREPARE_LIB_CMD)
prepare: PREPARE_MAN_CMD=$(DEFAULT_PREPARE_MAN_CMD)
prepare: prepare-base

prepare-base: PREPARE_SOURCE_DIR=$(PREPARE_HOST)/stage$(PREPARE_STAGE)
prepare-base: PREPARE_SOURCE_BIN_DIR=$(PREPARE_SOURCE_DIR)/bin
prepare-base: PREPARE_SOURCE_LIB_DIR=$(PREPARE_SOURCE_DIR)/$(CFG_LIBDIR_RELATIVE)
prepare-base: PREPARE_SOURCE_MAN_DIR=$(S)/man
prepare-base: PREPARE_DEST_BIN_DIR=$(PREPARE_DEST_DIR)/bin
prepare-base: PREPARE_DEST_LIB_DIR=$(PREPARE_DEST_DIR)/$(CFG_LIBDIR_RELATIVE)
prepare-base: PREPARE_DEST_MAN_DIR=$(PREPARE_DEST_DIR)/man1
prepare-base: prepare-host prepare-targets

prepare-everything: prepare-host prepare-targets

DEFAULT_PREPARE_DIR_CMD = umask 022 && mkdir -p
DEFAULT_PREPARE_BIN_CMD = install -m755
DEFAULT_PREPARE_LIB_CMD = install -m644
DEFAULT_PREPARE_MAN_CMD = install -m755

# On windows we install from stage3, but on unix only stage2
# Because of the way these rules are organized, preparing from any
# stage requires all these stages to be built
ifdef CFG_WINDOWSY_$(CFG_BUILD)
PREPARE_STAGES=3
else
PREPARE_STAGES=2
endif

# Create a directory
# $(1) is the directory
define PREPARE_DIR
	@$(Q)$(call E, install: $(1))
	$(Q)$(PREPARE_DIR_CMD) $(1)
endef

# Copy an executable
# $(1) is the filename/libname-glob
define PREPARE_BIN
	@$(call E, install: $(PREPARE_DEST_BIN_DIR)/$(1))
	$(Q)$(PREPARE_BIN_CMD) $(PREPARE_SOURCE_BIN_DIR)/$(1) $(PREPARE_DEST_BIN_DIR)/$(1)
endef

# Copy a dylib or rlib
# $(1) is the filename/libname-glob
#
# XXX: Don't remove the $(nop) command below!
# Yeah, that's right, it's voodoo. Something in the way this macro is being expanded
# causes it to parse incorrectly. Throwing in that empty command seems to fix the
# problem. I'm sorry, just don't remove the $(nop), alright?
define PREPARE_LIB
	$(nop)
	@$(call E, install: $(PREPARE_WORKING_DEST_LIB_DIR)/$(1))
	$(Q)LIB_NAME="$(notdir $(lastword $(wildcard $(PREPARE_WORKING_SOURCE_LIB_DIR)/$(1))))"; \
	MATCHES="$(filter-out %$(notdir $(lastword $(wildcard $(PREPARE_WORKING_SOURCE_LIB_DIR)/$(1)))),\
                        $(wildcard $(PREPARE_WORKING_DEST_LIB_DIR)/$(1)))"; \
	if [ -n "$$MATCHES" ]; then                                              \
	  echo "warning: one or libraries matching Rust library '$(1)'" &&       \
	  echo "  (other than '$$LIB_NAME' itself) already present"     &&       \
	  echo "  at destination $(PREPARE_WORKING_DEST_LIB_DIR):"                    &&       \
	  echo $$MATCHES ;                                                       \
	fi
	$(Q)$(PREPARE_LIB_CMD) `ls -drt1 $(PREPARE_WORKING_SOURCE_LIB_DIR)/$(1) | tail -1` $(PREPARE_WORKING_DEST_LIB_DIR)/
endef

# Copy a man page
# $(1) - source dir
define PREPARE_MAN
	@$(call E, install: $(PREPARE_DEST_MAN_DIR)/$(1))
	$(Q)$(PREPARE_MAN_CMD) $(PREPARE_SOURCE_MAN_DIR)/$(1) $(PREPARE_DEST_MAN_DIR)/$(1)
endef


PREPARE_TOOLS = $(filter-out compiletest, $(TOOLS))

prepare-host: prepare-host-tools

prepare-host-tools: \
        $(foreach tool, $(PREPARE_TOOLS),\
          $(foreach stage,$(PREPARE_STAGES),\
            $(foreach host,$(CFG_HOST),\
              prepare-host-tool-$(tool)-$(stage)-$(host))))

prepare-host-dirs:
	$(call PREPARE_DIR,$(PREPARE_DEST_BIN_DIR))
	$(call PREPARE_DIR,$(PREPARE_DEST_LIB_DIR))
	$(call PREPARE_DIR,$(PREPARE_DEST_MAN_DIR))

# $(1) is tool
# $(2) is stage
# $(3) is host
define DEF_PREPARE_HOST_TOOL
prepare-host-tool-$(1)-$(2)-$(3): $$(foreach dep,$$(TOOL_DEPS_$(1)),prepare-host-lib-$$(dep)-$(2)-$(3)) \
                                  $$(HBIN$(2)_H_$(3))/$(1)$$(X_$(3)) \
                                  prepare-host-dirs
	$$(if $$(findstring $(2), $$(PREPARE_STAGE)),\
      $$(if $$(findstring $(3), $$(PREPARE_HOST)),\
        $$(call PREPARE_BIN,$(1)$$(X_$$(PREPARE_HOST))),),)
	$$(if $$(findstring $(2), $$(PREPARE_STAGE)),\
      $$(if $$(findstring $(3), $$(PREPARE_HOST)),\
        $$(call PREPARE_MAN,$(1).1),),)
endef

$(foreach tool,$(PREPARE_TOOLS),\
  $(foreach stage,$(PREPARE_STAGES),\
    $(foreach host,$(CFG_HOST),\
        $(eval $(call DEF_PREPARE_HOST_TOOL,$(tool),$(stage),$(host))))))

# For host libraries only install dylibs, not rlibs since the host libs are only
# used to support rustc and rustc uses dynamic linking
#
# $(1) is tool
# $(2) is stage
# $(3) is host
define DEF_PREPARE_HOST_LIB
prepare-host-lib-$(1)-$(2)-$(3): PREPARE_WORKING_SOURCE_LIB_DIR=$$(PREPARE_SOURCE_LIB_DIR)
prepare-host-lib-$(1)-$(2)-$(3): PREPARE_WORKING_DEST_LIB_DIR=$$(PREPARE_DEST_LIB_DIR)
prepare-host-lib-$(1)-$(2)-$(3): $$(foreach dep,$$(RUST_DEPS_$(1)),prepare-host-lib-$$(dep)-$(2)-$(3))\
                                 $$(HLIB$(2)_H_$(3))/stamp.$(1) \
                                 prepare-host-dirs
	$$(if $$(findstring $(2), $$(PREPARE_STAGE)),\
      $$(if $$(findstring $(3), $$(PREPARE_HOST)),\
        $$(call PREPARE_LIB,$$(call CFG_LIB_GLOB_$$(PREPARE_HOST),$(1))),),)
endef

$(foreach lib,$(CRATES),\
  $(foreach stage,$(PREPARE_STAGES),\
    $(foreach host,$(CFG_HOST),\
      $(eval $(call DEF_PREPARE_HOST_LIB,$(lib),$(stage),$(host))))))

prepare-targets:\
        $(foreach host,$(CFG_HOST),\
           $(foreach target,$(CFG_TARGET),\
             $(foreach stage,$(PREPARE_STAGES),\
               prepare-target-$(target)-host-$(host)-$(stage))))

# $(1) is stage
# $(2) is target
# $(3) is host
define DEF_PREPARE_TARGET_N
# Rebind PREPARE_*_LIB_DIR to point to rustlib, then install the libs for the targets
prepare-target-$(2)-host-$(3)-$(1): PREPARE_WORKING_SOURCE_LIB_DIR=$$(PREPARE_SOURCE_LIB_DIR)/$$(CFG_RUSTLIBDIR)/$(2)/lib
prepare-target-$(2)-host-$(3)-$(1): PREPARE_WORKING_DEST_LIB_DIR=$$(PREPARE_DEST_LIB_DIR)/$$(CFG_RUSTLIBDIR)/$(2)/lib
prepare-target-$(2)-host-$(3)-$(1): \
        $$(foreach crate,$$(TARGET_CRATES), \
          $$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(crate)) \
        $$(if $$(findstring $(2),$$(CFG_HOST)), \
          $$(foreach crate,$$(HOST_CRATES), \
            $$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(crate)),)
# Only install if this host and target combo is being prepared
	$$(if $$(findstring $(1), $$(PREPARE_STAGE)),\
      $$(if $$(findstring $(2), $$(PREPARE_TARGETS)),\
        $$(if $$(findstring $(3), $$(PREPARE_HOST)),\
          $$(call PREPARE_DIR,$$(PREPARE_WORKING_DEST_LIB_DIR))\
          $$(foreach crate,$$(TARGET_CRATES),\
            $$(call PREPARE_LIB,$$(call CFG_LIB_GLOB_$(2),$$(crate)))\
            $$(call PREPARE_LIB,$$(call CFG_RLIB_GLOB,$$(crate))))\
          $$(if $$(findstring $(2),$$(CFG_HOST)),\
            $$(foreach crate,$$(HOST_CRATES),\
              $$(call PREPARE_LIB,$$(call CFG_LIB_GLOB_$(2),$$(crate)))\
              $$(call PREPARE_LIB,$$(call CFG_RLIB_GLOB,$$(crate)))),)\
          $$(call PREPARE_LIB,libmorestack.a) \
          $$(call PREPARE_LIB,libcompiler-rt.a),),),)
endef

$(foreach host,$(CFG_HOST),\
  $(foreach target,$(CFG_TARGET), \
    $(foreach stage,$(PREPARE_STAGES),\
      $(eval $(call DEF_PREPARE_TARGET_N,$(stage),$(target),$(host))))))
