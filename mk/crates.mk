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
#	Rust dependencies are listed bare (i.e. std) and native
#	dependencies have a "native:" prefix (i.e. native:hoedown). All deps
#	will be built before the crate itself is built.
#
#   TOOL_DEPS_<tool>/TOOL_SOURCE_<tool>
#	Similar to the DEPS variable, this is the library crate dependencies
#	list for tool as well as the source file for the specified tool
#
# You shouldn't need to modify much other than these variables. Crates are
# automatically generated for all stage/host/target combinations.
################################################################################

TARGET_CRATES := libc std flate arena term \
                 serialize getopts collections test rand \
                 log graphviz core rbml alloc \
                 rustc_unicode rustc_bitflags \
		 alloc_system
RUSTC_CRATES := rustc rustc_typeck rustc_mir rustc_borrowck rustc_resolve rustc_driver \
                rustc_trans rustc_back rustc_llvm rustc_privacy rustc_lint \
                rustc_data_structures rustc_front rustc_platform_intrinsics \
                rustc_plugin rustc_metadata
HOST_CRATES := syntax $(RUSTC_CRATES) rustdoc fmt_macros
TOOLS := compiletest rustdoc rustc rustbook error-index-generator

DEPS_core :=
DEPS_alloc := core libc alloc_system
DEPS_alloc_system := core libc
DEPS_collections := core alloc rustc_unicode
DEPS_libc := core
DEPS_rand := core
DEPS_rustc_bitflags := core
DEPS_rustc_unicode := core

DEPS_std := core libc rand alloc collections rustc_unicode \
	native:rust_builtin native:backtrace \
	alloc_system
DEPS_arena := std
DEPS_glob := std
DEPS_flate := std native:miniz
DEPS_fmt_macros = std
DEPS_getopts := std
DEPS_graphviz := std
DEPS_log := std
DEPS_num := std
DEPS_rbml := std log serialize
DEPS_serialize := std log
DEPS_term := std log
DEPS_test := std getopts serialize rbml term native:rust_test_helpers

DEPS_syntax := std term serialize log fmt_macros arena libc rustc_bitflags

DEPS_rustc := syntax flate arena serialize getopts rbml rustc_front\
              log graphviz rustc_llvm rustc_back rustc_data_structures
DEPS_rustc_back := std syntax rustc_llvm rustc_front flate log libc
DEPS_rustc_borrowck := rustc rustc_front log graphviz syntax
DEPS_rustc_data_structures := std log serialize
DEPS_rustc_driver := arena flate getopts graphviz libc rustc rustc_back rustc_borrowck \
                     rustc_typeck rustc_mir rustc_resolve log syntax serialize rustc_llvm \
	             rustc_trans rustc_privacy rustc_lint rustc_front rustc_plugin \
                     rustc_metadata
DEPS_rustc_front := std syntax log serialize
DEPS_rustc_lint := rustc log syntax
DEPS_rustc_llvm := native:rustllvm libc std rustc_bitflags
DEPS_rustc_metadata := rustc rustc_front syntax rbml
DEPS_rustc_mir := rustc rustc_front syntax
DEPS_rustc_resolve := rustc rustc_front log syntax
DEPS_rustc_platform_intrinsics := rustc rustc_llvm
DEPS_rustc_plugin := rustc rustc_metadata syntax
DEPS_rustc_privacy := rustc rustc_front log syntax
DEPS_rustc_trans := arena flate getopts graphviz libc rustc rustc_back rustc_mir \
                    log syntax serialize rustc_llvm rustc_front rustc_platform_intrinsics
DEPS_rustc_typeck := rustc syntax rustc_front rustc_platform_intrinsics

DEPS_rustdoc := rustc rustc_driver native:hoedown serialize getopts \
                test rustc_lint rustc_front


TOOL_DEPS_compiletest := test getopts
TOOL_DEPS_rustdoc := rustdoc
TOOL_DEPS_rustc := rustc_driver
TOOL_DEPS_rustbook := std rustdoc
TOOL_DEPS_error-index-generator := rustdoc syntax serialize
TOOL_SOURCE_compiletest := $(S)src/compiletest/compiletest.rs
TOOL_SOURCE_rustdoc := $(S)src/driver/driver.rs
TOOL_SOURCE_rustc := $(S)src/driver/driver.rs
TOOL_SOURCE_rustbook := $(S)src/rustbook/main.rs
TOOL_SOURCE_error-index-generator := $(S)src/error-index-generator/main.rs

ONLY_RLIB_core := 1
ONLY_RLIB_libc := 1
ONLY_RLIB_alloc := 1
ONLY_RLIB_rand := 1
ONLY_RLIB_collections := 1
ONLY_RLIB_rustc_unicode := 1
ONLY_RLIB_rustc_bitflags := 1
ONLY_RLIB_alloc_system := 1

# Documented-by-default crates
DOC_CRATES := std alloc collections core libc rustc_unicode

ifeq ($(CFG_DISABLE_JEMALLOC),)
TARGET_CRATES += alloc_jemalloc
DEPS_std += alloc_jemalloc
DEPS_alloc_jemalloc := core libc native:jemalloc
ONLY_RLIB_alloc_jemalloc := 1
else
RUSTFLAGS_rustc_back := --cfg disable_jemalloc
endif

################################################################################
# You should not need to edit below this line
################################################################################

CRATES := $(TARGET_CRATES) $(HOST_CRATES)

# This macro creates some simple definitions for each crate being built, just
# some munging of all of the parameters above.
#
# $(1) is the crate to generate variables for
define RUST_CRATE
CRATEFILE_$(1) := $$(SREL)src/lib$(1)/lib.rs
RSINPUTS_$(1) := $$(call rwildcard,$(S)src/lib$(1)/,*.rs)
RUST_DEPS_$(1) := $$(filter-out native:%,$$(DEPS_$(1)))
NATIVE_DEPS_$(1) := $$(patsubst native:%,%,$$(filter native:%,$$(DEPS_$(1))))
endef

$(foreach crate,$(CRATES),$(eval $(call RUST_CRATE,$(crate))))

# Similar to the macro above for crates, this macro is for tools
#
# $(1) is the crate to generate variables for
define RUST_TOOL
TOOL_INPUTS_$(1) := $$(call rwildcard,$$(dir $$(TOOL_SOURCE_$(1))),*.rs)
endef

$(foreach crate,$(TOOLS),$(eval $(call RUST_TOOL,$(crate))))

ifdef CFG_DISABLE_ELF_TLS
RUSTFLAGS_std := --cfg no_elf_tls
endif

CRATEFILE_libc := $(SREL)src/liblibc/src/lib.rs
RUSTFLAGS_libc := --cfg stdbuild
