# Copyright 2012 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

######################################################################
# rustc LLVM-extensions (C++) library variables and rules
######################################################################

define DEF_RUSTLLVM_TARGETS

# FIXME: Lately, on windows, llvm-config --includedir is not enough
# to find the llvm includes (probably because we're not actually installing
# llvm, but using it straight out of the build directory)
ifdef CFG_WINDOWSY_$(1)
LLVM_EXTRA_INCDIRS_$(1)= $$(call CFG_CC_INCLUDE_$(1),$(S)src/llvm/include) \
                         $$(call CFG_CC_INCLUDE_$(1),\
			   $$(CFG_LLVM_BUILD_DIR_$(1))/include)
endif

RUSTLLVM_OBJS_CS_$(1) := $$(addprefix rustllvm/, \
	RustWrapper.cpp PassWrapper.cpp \
	ArchiveWrapper.cpp)

RUSTLLVM_INCS_$(1) = $$(LLVM_EXTRA_INCDIRS_$(1)) \
                     $$(call CFG_CC_INCLUDE_$(1),$$(LLVM_INCDIR_$(1))) \
                     $$(call CFG_CC_INCLUDE_$(1),$$(S)src/rustllvm/include)
RUSTLLVM_OBJS_OBJS_$(1) := $$(RUSTLLVM_OBJS_CS_$(1):rustllvm/%.cpp=$(1)/rustllvm/%.o)

# Flag that we are building with Rust's llvm fork
ifeq ($(CFG_LLVM_ROOT),)
RUSTLLVM_CXXFLAGS_$(1) := -DLLVM_RUSTLLVM
endif

# Note that we appease `cl.exe` and its need for some sort of exception
# handling flag with the `EHsc` argument here as well.
ifeq ($$(findstring msvc,$(1)),msvc)
EXTRA_RUSTLLVM_CXXFLAGS_$(1) := //EHsc
endif

$$(RT_OUTPUT_DIR_$(1))/$$(call CFG_STATIC_LIB_NAME_$(1),rustllvm): \
	    $$(RUSTLLVM_OBJS_OBJS_$(1))
	@$$(call E, link: $$@)
	$$(Q)$$(call CFG_CREATE_ARCHIVE_$(1),$$@) $$^

RUSTLLVM_COMPONENTS_$(1) = $$(shell echo $$(LLVM_ALL_COMPONENTS_$(1)) |\
	tr 'a-z-' 'A-Z_'| sed -e 's/^ //;s/\([^ ]*\)/\-DLLVM_COMPONENT_\1/g')

# On MSVC we need to double-escape arguments that llvm-config printed which
# start with a '/'. The shell we're running in will auto-translate the argument
# `/foo` to `C:/msys64/foo` but we really want it to be passed through as `/foo`
# so the argument passed to our shell must be `//foo`.
$(1)/rustllvm/%.o: $(S)src/rustllvm/%.cpp $$(MKFILE_DEPS) $$(LLVM_CONFIG_$(1))
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_CXX_$(1), $$@,) \
		$$(subst  /,//,$$(LLVM_CXXFLAGS_$(1))) \
		$$(RUSTLLVM_COMPONENTS_$(1)) \
		$$(RUSTLLVM_CXXFLAGS_$(1)) \
		$$(EXTRA_RUSTLLVM_CXXFLAGS_$(1)) \
		$$(RUSTLLVM_INCS_$(1)) \
		$$<
endef

# Instantiate template for all stages
$(foreach host,$(CFG_HOST), \
 $(eval $(call DEF_RUSTLLVM_TARGETS,$(host))))
