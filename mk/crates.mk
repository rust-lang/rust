# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

################################################################################
# Rust's standard distribution of crates and tools
#
# The crates outlined below are the standard distribution of libraries provided
# in a rust installation. These rules are meant to abstract over the
# dependencies (both native and rust) of crates and basically generate all the
# necessary makefile rules necessary to build everything.
#
# Here's an explanation of the variables below
#
#   TARGET_CRATES
#	This list of crates will be built for all targets, including
#	cross-compiled targets
#
#   HOST_CRATES
#	This list of crates will be compiled for only host targets. Note that
#	this set is explicitly *not* a subset of TARGET_CRATES, but rather it is
#	a disjoint set. Nothing in the TARGET_CRATES set can depend on crates in
#	the HOST_CRATES set, but the HOST_CRATES set can depend on target
#	crates.
#
#   TOOLS
#	A list of all tools which will be built as part of the compilation
#	process. It is currently assumed that most tools are built through
#	src/driver/driver.rs with a particular configuration (there's a
#	corresponding library providing the implementation)
#
#   DEPS_<crate>
#	These lists are the dependencies of the <crate> that is to be built.
#	Rust dependencies are listed bare (i.e. std, extra, green) and native
#	dependencies have a "native:" prefix (i.e. native:sundown). All deps
#	will be built before the crate itself is built.
#
#   TOOL_DEPS_<tool>/TOOL_SOURCE_<tool>
#	Similar to the DEPS variable, this is the library crate dependencies
#	list for tool as well as the source file for the specified tool
#
# You shouldn't need to modify much other than these variables. Crates are
# automatically generated for all stage/host/target combinations.
################################################################################

TARGET_CRATES := std extra green rustuv native flate arena glob term semver uuid sync
HOST_CRATES := syntax rustc rustdoc
CRATES := $(TARGET_CRATES) $(HOST_CRATES)
TOOLS := compiletest rustdoc rustc

DEPS_std := native:rustrt
DEPS_extra := std term sync
DEPS_green := std
DEPS_rustuv := std native:uv native:uv_support
DEPS_native := std
DEPS_syntax := std extra term
DEPS_rustc := syntax native:rustllvm flate arena sync
DEPS_rustdoc := rustc native:sundown sync
DEPS_flate := std native:miniz
DEPS_arena := std extra
DEPS_glob := std
DEPS_term := std
DEPS_semver := std
DEPS_uuid := std extra
DEPS_sync := std

TOOL_DEPS_compiletest := extra green rustuv
TOOL_DEPS_rustdoc := rustdoc green rustuv
TOOL_DEPS_rustc := rustc green rustuv
TOOL_SOURCE_compiletest := $(S)src/compiletest/compiletest.rs
TOOL_SOURCE_rustdoc := $(S)src/driver/driver.rs
TOOL_SOURCE_rustc := $(S)src/driver/driver.rs

################################################################################
# You should not need to edit below this line
################################################################################

DOC_CRATES := $(filter-out rustc, $(filter-out syntax, $(CRATES)))

# This macro creates some simple definitions for each crate being built, just
# some munging of all of the parameters above.
#
# $(1) is the crate to generate variables for
define RUST_CRATE
CRATEFILE_$(1) := $$(S)src/lib$(1)/lib.rs
RSINPUTS_$(1) := $$(wildcard $$(addprefix $(S)src/lib$(1), \
				*.rs */*.rs */*/*.rs */*/*/*.rs))
RUST_DEPS_$(1) := $$(filter-out native:%,$$(DEPS_$(1)))
NATIVE_DEPS_$(1) := $$(patsubst native:%,%,$$(filter native:%,$$(DEPS_$(1))))
endef

$(foreach crate,$(CRATES),$(eval $(call RUST_CRATE,$(crate))))

# Similar to the macro above for crates, this macro is for tools
#
# $(1) is the crate to generate variables for
define RUST_TOOL
TOOL_INPUTS_$(1) := $$(wildcard $$(addprefix $(S)$$(dir $$(TOOL_SOURCE_$(1))), \
				*.rs */*.rs */*/*.rs */*/*/*.rs))
endef

$(foreach crate,$(TOOLS),$(eval $(call RUST_TOOL,$(crate))))
