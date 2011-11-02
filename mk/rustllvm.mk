######################################################################
# rustc LLVM-extensions (C++) library variables and rules
######################################################################

define DEF_RUSTLLVM_TARGETS

RUSTLLVM_OBJS_CS_$(1) := $$(addprefix rustllvm/, RustGCMetadataPrinter.cpp \
    RustGCStrategy.cpp RustWrapper.cpp)

# Behind an ifdef for now since this requires a patched LLVM.
ifdef CFG_STACK_GROWTH
RUSTLLVM_OBJS_CS_$(1) += rustllvm/RustPrologHook.cpp
endif

RUSTLLVM_DEF_$(1) := rustllvm/rustllvm$$(CFG_DEF_SUFFIX)

RUSTLLVM_INCS_$(1) := -iquote $$(LLVM_INCDIR) \
                      -iquote $$(S)src/rustllvm/include
RUSTLLVM_OBJS_OBJS_$(1) := $$(RUSTLLVM_OBJS_CS_$(1):rustllvm/%.cpp=rustllvm/$(1)/%.o)

rustllvm/$(1)/$(CFG_RUSTLLVM): $$(RUSTLLVM_OBJS_OBJS_$(1)) \
                          $$(MKFILES) $$(RUSTLLVM_DEF_$(1))
	@$$(call E, link: $$@)
	$$(Q)$$(call CFG_LINK_C_$(1),$$@,$$(RUSTLLVM_OBJS_OBJS_$(1)) \
	  $$(CFG_GCCISH_PRE_LIB_FLAGS) $$(LLVM_LIBS) \
          $$(CFG_GCCISH_POST_LIB_FLAGS) \
          $$(LLVM_LDFLAGS),$$(RUSTLLVM_DEF_$(1)),$$(CFG_RUSTLLVM))

rustllvm/$(1)/%.o: rustllvm/%.cpp $$(MKFILES) \
		$$(CFG_LLVM_INST_DIR)/bin/llvm-config
	@$$(call E, compile: $$@)
	$$(Q)$$(call CFG_COMPILE_C_$(1), $$@, $$(LLVM_CXXFLAGS) $$(RUSTLLVM_INCS_$(1))) $$<
endef

# Instantiate template for all stages
$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call DEF_RUSTLLVM_TARGETS,$(target))))
