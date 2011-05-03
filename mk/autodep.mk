######################################################################
# Auto-dependency
######################################################################

ML_DEPFILES := $(BOOT_MLS:%.ml=%.d)
C_DEPFILES := $(RUNTIME_CS:%.cpp=%.d) $(RUSTLLVM_LIB_CS:%.cpp=%.d) \
              $(RUSTLLVM_OBJS_CS:%.cpp=%.d)

rt/%.d: rt/%.cpp $(MKFILES)
	@$(call E, dep: $@)
	$(Q)$(call CFG_DEPEND_C, $@ \
      $(subst $(S)src/,,$(patsubst %.cpp, %.o, $<)), \
      $(RUNTIME_INCS)) $< >$@.tmp
	$(Q)$(CFG_PATH_MUNGE) $@.tmp
	$(Q)rm -f $@.tmp.bak
	$(Q)mv $@.tmp $@

rustllvm/%.d: rustllvm/%.cpp $(MKFILES)
	@$(call E, dep: $@)
	$(Q)$(call CFG_DEPEND_C, $@ \
      $(subst $(S)src/,,$(patsubst %.cpp, %.o, $<)), \
      $(CFG_LLVM_CXXFLAGS) $(RUSTLLVM_INCS)) $< >$@.tmp
	$(Q)$(CFG_PATH_MUNGE) $@.tmp
	$(Q)rm -f $@.tmp.bak
	$(Q)mv $@.tmp $@

%.d: %.ml $(MKFILES)
	@$(call E, dep: $@)
	$(Q)ocamldep$(OPT) -slash $(BOOT_ML_DEP_INCS) $< >$@.tmp
	$(Q)$(CFG_PATH_MUNGE) $@.tmp
	$(Q)rm -f $@.tmp.bak
	$(Q)perl -i.bak -pe "s@$(S)src/@@go" $@.tmp
	$(Q)rm -f $@.tmp.bak
	$(Q)mv $@.tmp $@

%.d: %.mli $(MKFILES)
	@$(call E, dep: $@)
	$(Q)ocamldep$(OPT) -slash $(BOOT_ML_DEP_INCS) $< >$@.tmp
	$(Q)$(CFG_PATH_MUNGE) $@.tmp
	$(Q)rm -f $@.tmp.bak
	$(Q)perl -i.bak -pe "s@$(S)src/@@go" $@.tmp
	$(Q)rm -f $@.tmp.bak
	$(Q)mv $@.tmp $@

ifneq ($(MAKECMDGOALS),clean)
-include $(ML_DEPFILES) $(C_DEPFILES)
endif

RUSTBOOT_PROBE := $(wildcard boot/rustboot$(X))

ifneq ($(RUSTBOOT_PROBE),)
CFG_INFO := $(info cfg: using built boot/rustboot$(X) for rust deps)
CRATE_DEPFILES := $(subst $(S)src/,,$(ALL_TEST_CRATES:%.rc=%.d)) \
                  boot/$(CFG_STDLIB).d \
                  stage0/rustc$(X).d \
                  stage0/$(CFG_STDLIB).d

boot/$(CFG_STDLIB).d: $(STDLIB_CRATE) $(STDLIB_INPUTS) \
                      $(MKFILES) boot/rustboot$(X)
	@$(call E, dep: $@)
	$(BOOT) -o $(patsubst %.d,%$(X),$@) -shared -rdeps $< >$@.tmp
	$(Q)$(CFG_PATH_MUNGE) $@.tmp
	$(Q)rm -f $@.tmp.bak
	$(Q)mv $@.tmp $@

stage0/rustc$(X).d: $(COMPILER_CRATE) $(COMPILER_INPUTS) \
                    $(STDLIB_CRATE) $(MKFILES) boot/rustboot$(X)
	@$(call E, dep: $@)
	$(Q)touch $@

%.d: %.rc $(MKFILES)
	@$(call E, dep: $@)
	$(Q)touch $@

ifneq ($(MAKECMDGOALS),clean)
-include $(CRATE_DEPFILES)
endif
endif

depend: boot/rustboot$(X) $(CRATE_DEPFILES) $(ML_DEPFILES) $(C_DEPFILES)
