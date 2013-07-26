# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# Rules for non-core tools built with the compiler, both for target
# and host architectures

# The test runner that runs the cfail/rfail/rpass and bxench tests
COMPILETEST_CRATE := $(S)src/compiletest/compiletest.rs
COMPILETEST_INPUTS := $(wildcard $(S)src/compiletest/*.rs)

# Rustpkg, the package manager and build system
RUSTPKG_LIB := $(S)src/librustpkg/rustpkg.rs
RUSTPKG_INPUTS := $(wildcard $(S)src/librustpkg/*.rs)

# Rustdoc, the documentation tool
RUSTDOC_LIB := $(S)src/librustdoc/rustdoc.rs
RUSTDOC_INPUTS := $(wildcard $(S)src/librustdoc/*.rs)

# Rusti, the JIT REPL
RUSTI_LIB := $(S)src/librusti/rusti.rs
RUSTI_INPUTS := $(wildcard $(S)src/librusti/*.rs)

# Rust, the convenience tool
RUST_LIB := $(S)src/librust/rust.rs
RUST_INPUTS := $(wildcard $(S)src/librust/*.rs)

# FIXME: These are only built for the host arch. Eventually we'll
# have tools that need to built for other targets.
define TOOLS_STAGE_N_TARGET

$$(TBIN$(1)_T_$(4)_H_$(3))/compiletest$$(X_$(4)):			\
		$$(COMPILETEST_CRATE) $$(COMPILETEST_INPUTS)	\
		$$(SREQ$(1)_T_$(4)_H_$(3))			\
		| $$(TBIN$(1)_T_$(4)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$<

$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTPKG_$(4)):		\
		$$(RUSTPKG_LIB) $$(RUSTPKG_INPUTS)		    \
		$$(SREQ$(1)_T_$(4)_H_$(3))			\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTC_$(4)) \
		| $$(TLIB$(1)_T_$(4)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTPKG_GLOB_$(4)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(4)_H_$(3)) $$(WFLAGS_ST$(1)) -o $$@ $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTPKG_GLOB_$(4)),$$(notdir $$@))

$$(TBIN$(1)_T_$(4)_H_$(3))/rustpkg$$(X_$(4)):				\
		$$(DRIVER_CRATE) 							\
		$$(TSREQ$(1)_T_$(4)_H_$(3))				\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTPKG_$(4))	\
		| $$(TBIN$(1)_T_$(4)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) --cfg rustpkg -o $$@ $$<

$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTDOC_$(4)):		\
		$$(RUSTDOC_LIB) $$(RUSTDOC_INPUTS)			\
		$$(SREQ$(1)_T_$(4)_H_$(3))			\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTC_$(4)) \
		| $$(TLIB$(1)_T_$(4)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTDOC_GLOB_$(4)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTDOC_GLOB_$(4)),$$(notdir $$@))

$$(TBIN$(1)_T_$(4)_H_$(3))/rustdoc$$(X_$(4)):			\
		$$(DRIVER_CRATE) 							\
		$$(TSREQ$(1)_T_$(4)_H_$(3))						\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTDOC_$(4))			\
		| $$(TBIN$(1)_T_$(4)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) --cfg rustdoc -o $$@ $$<

$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTI_$(4)):		\
		$$(RUSTI_LIB) $$(RUSTI_INPUTS)			\
		$$(SREQ$(1)_T_$(4)_H_$(3))			\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTC_$(4))	\
		| $$(TLIB$(1)_T_$(4)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTI_GLOB_$(4)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTI_GLOB_$(4)),$$(notdir $$@))

$$(TBIN$(1)_T_$(4)_H_$(3))/rusti$$(X_$(4)):			\
		$$(DRIVER_CRATE) 							\
		$$(TSREQ$(1)_T_$(4)_H_$(3))			\
		$$(TLIB$(1)_T_$(4)_H_$(4))/$(CFG_LIBRUSTI_$(4)) \
		| $$(TBIN$(1)_T_$(4)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) --cfg rusti -o $$@ $$<

$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUST_$(4)):		\
		$$(RUST_LIB) $$(RUST_INPUTS)			\
		$$(SREQ$(1)_T_$(4)_H_$(3))				\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTPKG_$(4))	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTI_$(4))		\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTDOC_$(4))	\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTC_$(4))		\
		| $$(TLIB$(1)_T_$(4)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUST_GLOB_$(4)),$$(notdir $$@))
	$$(STAGE$(1)_T_$(4)_H_$(3)) -o $$@ $$< && touch $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUST_GLOB_$(4)),$$(notdir $$@))

$$(TBIN$(1)_T_$(4)_H_$(3))/rust$$(X_$(4)):			\
		$$(DRIVER_CRATE) 							\
		$$(TSREQ$(1)_T_$(4)_H_$(3))			\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUST_$(4)) \
		| $$(TBIN$(1)_T_$(4)_H_$(3))/
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(4)_H_$(3)) --cfg rust -o $$@ $$<

endef

define TOOLS_STAGE_N_HOST

$$(HBIN$(2)_H_$(4))/compiletest$$(X_$(4)):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/compiletest$$(X_$(4))	\
		$$(HSREQ$(2)_H_$(4))					\
		| $$(HBIN$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@


$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTPKG_$(4)):				\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTPKG_$(4))	\
		$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTC_$(4))		\
		$$(HSREQ$(2)_H_$(4))					\
		| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTPKG_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTPKG_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTPKG_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTPKG_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))

$$(HBIN$(2)_H_$(4))/rustpkg$$(X_$(4)):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/rustpkg$$(X_$(4))	\
		$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTPKG_$(4))	\
		$$(HSREQ$(2)_H_$(4))				\
		| $$(HBIN$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTDOC_$(4)):					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTDOC_$(4))	\
		$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTC_$(4))			\
		$$(HSREQ$(2)_H_$(4)) \
		| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTDOC_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTDOC_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTDOC_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTDOC_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))

$$(HBIN$(2)_H_$(4))/rustdoc$$(X_$(4)):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/rustdoc$$(X_$(4))	\
		$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTDOC_$(4))	\
		$$(HSREQ$(2)_H_$(4))				\
		| $$(HBIN$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTI_$(4)):					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUSTI_$(4))	\
		$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTC_$(4))			\
		$$(HSREQ$(2)_H_$(4)) \
		| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTI_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUSTI_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTI_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUSTI_DSYM_GLOB_$(4))) \
	        $$(HLIB$(2)_H_$(4))

$$(HBIN$(2)_H_$(4))/rusti$$(X_$(4)):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/rusti$$(X_$(4))	\
		$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTI_$(4))	\
		$$(HSREQ$(2)_H_$(4))				\
		| $$(HBIN$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUST_$(4)):					\
		$$(TLIB$(1)_T_$(4)_H_$(3))/$(CFG_LIBRUST_$(4))	\
		$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUSTC_$(4))	\
		$$(HSREQ$(2)_H_$(4))				\
		| $$(HLIB$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(call REMOVE_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUST_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp $$< $$@
	$$(call LIST_ALL_OLD_GLOB_MATCHES_EXCEPT,$$(dir $$@),$(LIBRUST_GLOB_$(4)),$$(notdir $$@))
	$$(Q)cp -R $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUST_GLOB_$(4)) \
		$$(wildcard $$(TLIB$(1)_T_$(4)_H_$(3))/$(LIBRUST_DSYM_GLOB)_$(4)) \
	        $$(HLIB$(2)_H_$(4))

$$(HBIN$(2)_H_$(4))/rust$$(X_$(4)):				\
		$$(TBIN$(1)_T_$(4)_H_$(3))/rust$$(X_$(4))	\
		$$(HLIB$(2)_H_$(4))/$(CFG_LIBRUST_$(4))	\
		$$(HSREQ$(2)_H_$(4))				\
		| $$(HBIN$(2)_H_$(4))/
	@$$(call E, cp: $$@)
	$$(Q)cp $$< $$@

endef

$(foreach host,$(CFG_HOST_TRIPLES),				\
$(foreach target,$(CFG_TARGET_TRIPLES),				\
 $(eval $(call TOOLS_STAGE_N_TARGET,0,1,$(host),$(target)))	\
 $(eval $(call TOOLS_STAGE_N_TARGET,1,2,$(host),$(target)))	\
 $(eval $(call TOOLS_STAGE_N_TARGET,2,3,$(host),$(target)))	\
 $(eval $(call TOOLS_STAGE_N_TARGET,3,bogus,$(host),$(target)))))

$(foreach host,$(CFG_HOST_TRIPLES),				\
 $(eval $(call TOOLS_STAGE_N_HOST,0,1,$(host),$(host)))	\
 $(eval $(call TOOLS_STAGE_N_HOST,1,2,$(host),$(host)))	\
 $(eval $(call TOOLS_STAGE_N_HOST,2,3,$(host),$(host))))
