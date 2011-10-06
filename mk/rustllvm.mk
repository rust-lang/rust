######################################################################
# rustc LLVM-extensions (C++) library variables and rules
######################################################################

RUSTLLVM_OBJS_CS := $(addprefix rustllvm/, RustGCMetadataPrinter.cpp \
    RustGCStrategy.cpp RustWrapper.cpp)

RUSTLLVM_DEF := rustllvm/rustllvm$(CFG_DEF_SUFFIX)

RUSTLLVM_INCS := -iquote $(CFG_LLVM_INCDIR) \
                 -iquote $(S)src/rustllvm/include
RUSTLLVM_OBJS_OBJS := $(RUSTLLVM_OBJS_CS:.cpp=.o)

rustllvm/$(CFG_RUSTLLVM): $(RUSTLLVM_OBJS_OBJS) \
                          $(MKFILES) $(RUSTLLVM_DEF)
	@$(call E, link: $@)
	$(Q)$(call CFG_LINK_C,$@,$(RUSTLLVM_OBJS_OBJS) \
	  $(CFG_GCCISH_PRE_LIB_FLAGS) $(CFG_LLVM_LIBS) \
          $(CFG_GCCISH_POST_LIB_FLAGS) \
          $(CFG_LLVM_LDFLAGS),$(RUSTLLVM_DEF),$(CFG_RUSTLLVM))

rustllvm/%.o: rustllvm/%.cpp $(MKFILES)
	@$(call E, compile: $@)
	$(Q)$(call CFG_COMPILE_C, $@, $(CFG_LLVM_CXXFLAGS) $(RUSTLLVM_INCS)) $<

