######################################################################
# rustc LLVM-extensions (C++) library variables and rules
######################################################################

define DEF_RUSTLLVM_TARGETS

# FIXME: Lately, on windows, llvm-config --includedir is not enough
# to find the llvm includes (probably because we're not actually installing
# llvm, but using it straight out of the build directory)
ifdef CFG_WINDOWSY
LLVM_EXTRA_INCDIRS_$(1)= -iquote $(S)src/llvm/include \
                         -iquote llvm/$(1)/include
endif

RUSTLLVM_OBJS_CS_$(1) := $$(addprefix rustllvm/, RustGCMetadataPrinter.cpp \
    RustGCStrategy.cpp RustWrapper.cpp)

# Behind an ifdef for now since this requires a patched LLVM.
ifdef CFG_STACK_GROWTH
RUSTLLVM_OBJS_CS_$(1) += rustllvm/RustPrologHook.cpp
endif

RUSTLLVM_DEF_$(1) := rustllvm/rustllvm$$(CFG_DEF_SUFFIX)

RUSTLLVM_INCS_$(1) = $$(LLVM_EXTRA_INCDIRS_$(1)) \
                     -iquote $$(LLVM_INCDIR_$(1)) \
                     -iquote $$(S)src/rustllvm/include
RUSTLLVM_OBJS_OBJS_$(1) := $$(RUSTLLVM_OBJS_CS_$(1):rustllvm/%.cpp=rustllvm/$(1)/%.o)

rustllvm/$(1)/$(CFG_RUSTLLVM): $$(RUSTLLVM_OBJS_OBJS_$(1)) \
                          $$(MKFILE_DEPS) $$(RUSTLLVM_DEF_$(1))
	@$$(call E, link: $$@)
	$$(Q)$$(call CFG_LINK_C_$(1),$$@,$$(RUSTLLVM_OBJS_OBJS_$(1)) \
	  $$(CFG_GCCISH_PRE_LIB_FLAGS) $$(LLVM_LIBS_$(1)) \
          $$(CFG_GCCISH_POST_LIB_FLAGS) \
          $$(LLVM_LDFLAGS_$(1)),$$(RUSTLLVM_DEF_$(1)),$$(CFG_RUSTLLVM))

rustllvm/$(1)/%.o: rustllvm/%.cpp $$(MKFILE_DEPS) $$(LLVM_CONFIG_$(1))
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, $$(LLVM_CXXFLAGS_$(1)) $$(RUSTLLVM_INCS_$(1))) $$<
endef

# Instantiate template for all stages
$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call DEF_RUSTLLVM_TARGETS,$(target))))
