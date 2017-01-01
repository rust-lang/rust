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

TARGET_CRATES := libc std term \
                 getopts collections test rand \
                 compiler_builtins core alloc \
                 std_unicode rustc_bitflags \
		 alloc_system alloc_jemalloc \
		 panic_abort panic_unwind unwind rustc_i128
RUSTC_CRATES := rustc rustc_typeck rustc_mir rustc_borrowck rustc_resolve rustc_driver \
                rustc_trans rustc_back rustc_llvm rustc_privacy rustc_lint \
                rustc_data_structures rustc_platform_intrinsics rustc_errors \
                rustc_plugin rustc_metadata rustc_passes rustc_save_analysis \
                rustc_const_eval rustc_const_math rustc_incremental proc_macro
HOST_CRATES := syntax syntax_ext proc_macro_tokens proc_macro_plugin syntax_pos $(RUSTC_CRATES) \
		rustdoc fmt_macros flate arena graphviz log serialize
TOOLS := compiletest rustdoc rustc rustbook error_index_generator

DEPS_core :=
DEPS_compiler_builtins := core native:compiler-rt
DEPS_alloc := core libc alloc_system
DEPS_alloc_system := core libc
DEPS_alloc_jemalloc := core libc native:jemalloc
DEPS_collections := core alloc std_unicode
DEPS_libc := core
DEPS_rand := core
DEPS_rustc_bitflags := core
DEPS_std_unicode := core
DEPS_panic_abort := libc alloc
DEPS_panic_unwind := libc alloc unwind
DEPS_unwind := libc

RUSTFLAGS_compiler_builtins := -lstatic=compiler-rt
RUSTFLAGS_panic_abort := -C panic=abort

DEPS_std := core libc rand alloc collections compiler_builtins std_unicode \
	native:backtrace \
	alloc_system panic_abort panic_unwind unwind
DEPS_arena := std
DEPS_glob := std
DEPS_flate := std native:miniz
DEPS_fmt_macros = std
DEPS_getopts := std
DEPS_graphviz := std
DEPS_log := std
DEPS_num := std
DEPS_serialize := std log rustc_i128
DEPS_term := std
DEPS_test := std getopts term native:rust_test_helpers
DEPS_rustc_i128 = std

DEPS_syntax := std term serialize log arena libc rustc_bitflags std_unicode rustc_errors \
			syntax_pos rustc_data_structures rustc_i128
DEPS_syntax_ext := syntax syntax_pos rustc_errors fmt_macros proc_macro
DEPS_proc_macro := syntax syntax_pos rustc_plugin log
DEPS_syntax_pos := serialize
DEPS_proc_macro_tokens := syntax syntax_pos log
DEPS_proc_macro_plugin := syntax syntax_pos rustc_plugin log proc_macro_tokens

DEPS_rustc_const_math := std syntax log serialize rustc_i128
DEPS_rustc_const_eval := rustc_const_math rustc syntax log serialize \
			     rustc_back graphviz syntax_pos rustc_i128

DEPS_rustc := syntax fmt_macros flate arena serialize getopts \
              log graphviz rustc_llvm rustc_back rustc_data_structures\
	      rustc_const_math syntax_pos rustc_errors rustc_i128
DEPS_rustc_back := std syntax flate log libc
DEPS_rustc_borrowck := rustc log graphviz syntax syntax_pos rustc_errors rustc_mir
DEPS_rustc_data_structures := std log serialize libc
DEPS_rustc_driver := arena flate getopts graphviz libc rustc rustc_back rustc_borrowck \
                     rustc_typeck rustc_mir rustc_resolve log syntax serialize rustc_llvm \
                     rustc_trans rustc_privacy rustc_lint rustc_plugin \
                     rustc_metadata syntax_ext proc_macro_plugin \
                     rustc_passes rustc_save_analysis rustc_const_eval \
                     rustc_incremental syntax_pos rustc_errors proc_macro rustc_data_structures
DEPS_rustc_errors := log libc serialize syntax_pos
DEPS_rustc_lint := rustc log syntax syntax_pos rustc_const_eval rustc_i128
DEPS_rustc_llvm := native:rustllvm libc std rustc_bitflags
DEPS_proc_macro := std syntax
DEPS_rustc_metadata := rustc syntax syntax_pos rustc_errors rustc_const_math \
			proc_macro syntax_ext rustc_i128
DEPS_rustc_passes := syntax syntax_pos rustc core rustc_const_eval rustc_errors
DEPS_rustc_mir := rustc syntax syntax_pos rustc_const_math rustc_const_eval rustc_bitflags \
					rustc_i128
DEPS_rustc_resolve := arena rustc log syntax syntax_pos rustc_errors
DEPS_rustc_platform_intrinsics := std
DEPS_rustc_plugin := rustc rustc_metadata syntax syntax_pos rustc_errors
DEPS_rustc_privacy := rustc log syntax syntax_pos
DEPS_rustc_trans := arena flate getopts graphviz libc rustc rustc_back \
                    log syntax serialize rustc_llvm rustc_platform_intrinsics rustc_i128 \
                    rustc_const_math rustc_const_eval rustc_incremental rustc_errors syntax_pos
DEPS_rustc_incremental := rustc syntax_pos serialize rustc_data_structures
DEPS_rustc_save_analysis := rustc log syntax syntax_pos serialize
DEPS_rustc_typeck := rustc syntax syntax_pos rustc_platform_intrinsics rustc_const_math \
                     rustc_const_eval rustc_errors rustc_data_structures

DEPS_rustdoc := rustc rustc_driver native:hoedown serialize getopts test \
                rustc_lint rustc_const_eval syntax_pos rustc_data_structures

TOOL_DEPS_compiletest := test getopts log serialize
TOOL_DEPS_rustdoc := rustdoc
TOOL_DEPS_rustc := rustc_driver
TOOL_DEPS_rustbook := std rustdoc
TOOL_DEPS_error_index_generator := rustdoc syntax serialize
TOOL_SOURCE_compiletest := $(S)src/tools/compiletest/src/main.rs
TOOL_SOURCE_rustdoc := $(S)src/driver/driver.rs
TOOL_SOURCE_rustc := $(S)src/driver/driver.rs
TOOL_SOURCE_rustbook := $(S)src/tools/rustbook/main.rs
TOOL_SOURCE_error_index_generator := $(S)src/tools/error_index_generator/main.rs

ONLY_RLIB_compiler_builtins := 1
ONLY_RLIB_core := 1
ONLY_RLIB_libc := 1
ONLY_RLIB_alloc := 1
ONLY_RLIB_rand := 1
ONLY_RLIB_collections := 1
ONLY_RLIB_std_unicode := 1
ONLY_RLIB_rustc_i128 := 1
ONLY_RLIB_rustc_bitflags := 1
ONLY_RLIB_alloc_system := 1
ONLY_RLIB_alloc_jemalloc := 1
ONLY_RLIB_panic_unwind := 1
ONLY_RLIB_panic_abort := 1
ONLY_RLIB_unwind := 1

TARGET_SPECIFIC_alloc_jemalloc := 1

# Documented-by-default crates
DOC_CRATES := std alloc collections core libc std_unicode

ifeq ($(CFG_DISABLE_JEMALLOC),)
RUSTFLAGS_rustc_back := --cfg 'feature="jemalloc"'
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
NATIVE_DEPS_$(1) := $$(patsubst native:%,%,$$(filter native:%,$$(DEPS_$(1))))
endef

$(foreach crate,$(CRATES),$(eval $(call RUST_CRATE,$(crate))))

# $(1) - crate
# $(2) - target
define RUST_CRATE_DEPS
RUST_DEPS_$(1)_T_$(2) := $$(filter-out native:%,$$(DEPS_$(1)))
endef

$(foreach target,$(CFG_TARGET),\
 $(foreach crate,$(CRATES),$(eval $(call RUST_CRATE_DEPS,$(crate),$(target)))))

# $(1) - target
# $(2) - crate
define DEFINE_TARGET_CRATES
ifndef TARGET_SPECIFIC_$(2)
TARGET_CRATES_$(1) += $(2)
endif
endef

$(foreach target,$(CFG_TARGET),\
 $(foreach crate,$(TARGET_CRATES),\
  $(eval $(call DEFINE_TARGET_CRATES,$(target),$(crate)))))

# Similar to the macro above for crates, this macro is for tools
#
# $(1) is the crate to generate variables for
define RUST_TOOL
TOOL_INPUTS_$(1) := $$(call rwildcard,$$(dir $$(TOOL_SOURCE_$(1))),*.rs)
endef

$(foreach crate,$(TOOLS),$(eval $(call RUST_TOOL,$(crate))))

CRATEFILE_libc := $(SREL)src/liblibc/src/lib.rs
RUSTFLAGS_libc := --cfg stdbuild
