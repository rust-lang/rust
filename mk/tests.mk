######################################################################
# Testing variables
######################################################################

ALL_TEST_INPUTS = $(wildcard $(S)src/test/*/*.rs   \
                              $(S)src/test/*/*/*.rs \
                              $(S)src/test/*/*.rc)

ifneq ($(findstring check,$(MAKECMDGOALS)),)
XFAIL_INPUTS := $(shell grep -l xfail $(ALL_TEST_INPUTS))
TEST_XFAILS_STAGE0 := $(shell grep -l xfail-stage0 $(XFAIL_INPUTS))
TEST_XFAILS_STAGE1 := $(shell grep -l xfail-stage1 $(XFAIL_INPUTS))
TEST_XFAILS_STAGE2 := $(shell grep -l xfail-stage2 $(XFAIL_INPUTS))

ifdef MINGW_CROSS
TEST_XFAILS_STAGE0 += $(S)src/test/run-pass/native-mod.rc
TEST_XFAILS_STAGE1 += $(S)src/test/run-pass/native-mod.rc
TEST_XFAILS_STAGE2 += $(S)src/test/run-pass/native-mod.rc
endif
ifdef CFG_WINDOWSY
TEST_XFAILS_STAGE0 += $(S)src/test/run-pass/native-mod.rc
TEST_XFAILS_STAGE1 += $(S)src/test/run-pass/native-mod.rc
TEST_XFAILS_STAGE2 += $(S)src/test/run-pass/native-mod.rc
endif
endif


BENCH_RS := $(wildcard $(S)src/test/bench/shootout/*.rs) \
            $(wildcard $(S)src/test/bench/99-bottles/*.rs)
RPASS_RC := $(wildcard $(S)src/test/run-pass/*.rc)
RPASS_RS := $(wildcard $(S)src/test/run-pass/*.rs) $(BENCH_RS)
RFAIL_RC := $(wildcard $(S)src/test/run-fail/*.rc)
RFAIL_RS := $(wildcard $(S)src/test/run-fail/*.rs)
CFAIL_RC := $(wildcard $(S)src/test/compile-fail/*.rc)
CFAIL_RS := $(wildcard $(S)src/test/compile-fail/*.rs)

ifdef CHECK_XFAILS
TEST_RPASS_CRATES_STAGE0 := $(filter $(TEST_XFAILS_STAGE0), $(RPASS_RC))
TEST_RPASS_CRATES_STAGE1 := $(filter $(TEST_XFAILS_STAGE1), $(RPASS_RC))
TEST_RPASS_CRATES_STAGE2 := $(filter $(TEST_XFAILS_STAGE2), $(RPASS_RC))
TEST_RPASS_SOURCES_STAGE0 := $(filter $(TEST_XFAILS_STAGE0), $(RPASS_RS))
TEST_RPASS_SOURCES_STAGE1 := $(filter $(TEST_XFAILS_STAGE1), $(RPASS_RS))
TEST_RPASS_SOURCES_STAGE2 := $(filter $(TEST_XFAILS_STAGE2), $(RPASS_RS))
else
TEST_RPASS_CRATES_STAGE0 := $(filter-out $(TEST_XFAILS_STAGE0), $(RPASS_RC))
TEST_RPASS_CRATES_STAGE1 := $(filter-out $(TEST_XFAILS_STAGE1), $(RPASS_RC))
TEST_RPASS_CRATES_STAGE1 := $(filter-out $(TEST_XFAILS_STAGE2), $(RPASS_RC))
TEST_RPASS_SOURCES_STAGE0 := $(filter-out $(TEST_XFAILS_STAGE0), $(RPASS_RS))
TEST_RPASS_SOURCES_STAGE1 := $(filter-out $(TEST_XFAILS_STAGE1), $(RPASS_RS))
TEST_RPASS_SOURCES_STAGE2 := $(filter-out $(TEST_XFAILS_STAGE2), $(RPASS_RS))
endif

TEST_RPASS_EXES_STAGE0 := \
  $(subst $(S)src/,,$(TEST_RPASS_CRATES_STAGE0:.rc=.stage0$(X))) \
  $(subst $(S)src/,,$(TEST_RPASS_SOURCES_STAGE0:.rs=.stage0$(X)))
TEST_RPASS_EXES_STAGE1 := \
  $(subst $(S)src/,,$(TEST_RPASS_CRATES_STAGE1:.rc=.stage1$(X))) \
  $(subst $(S)src/,,$(TEST_RPASS_SOURCES_STAGE1:.rs=.stage1$(X)))
TEST_RPASS_EXES_STAGE2 := \
  $(subst $(S)src/,,$(TEST_RPASS_CRATES_STAGE1:.rc=.stage2$(X))) \
  $(subst $(S)src/,,$(TEST_RPASS_SOURCES_STAGE1:.rs=.stage2$(X)))

TEST_RPASS_OUTS_STAGE0 := \
  $(TEST_RPASS_EXES_STAGE0:.stage0$(X)=.stage0.out)
TEST_RPASS_OUTS_STAGE1 := \
  $(TEST_RPASS_EXES_STAGE1:.stage1$(X)=.stage1.out)
TEST_RPASS_OUTS_STAGE2 := \
  $(TEST_RPASS_EXES_STAGE2:.stage2$(X)=.stage2.out)

TEST_RPASS_TMPS_STAGE0 := \
  $(TEST_RPASS_EXES_STAGE0:.stage0$(X)=.stage0$(X).tmp)
TEST_RPASS_TMPS_STAGE1 := \
  $(TEST_RPASS_EXES_STAGE1:.stage1$(X)=.stage1$(X).tmp)
TEST_RPASS_TMPS_STAGE2 := \
  $(TEST_RPASS_EXES_STAGE2:.stage2$(X)=.stage2$(X).tmp)


TEST_RFAIL_CRATES_STAGE0 := $(filter-out $(TEST_XFAILS_STAGE0), $(RFAIL_RC))
TEST_RFAIL_CRATES_STAGE1 := $(filter-out $(TEST_XFAILS_STAGE1), $(RFAIL_RC))
TEST_RFAIL_CRATES_STAGE2 := $(filter-out $(TEST_XFAILS_STAGE2), $(RFAIL_RC))
TEST_RFAIL_SOURCES_STAGE0 := $(filter-out $(TEST_XFAILS_STAGE0), $(RFAIL_RS))
TEST_RFAIL_SOURCES_STAGE1 := $(filter-out $(TEST_XFAILS_STAGE1), $(RFAIL_RS))
TEST_RFAIL_SOURCES_STAGE2 := $(filter-out $(TEST_XFAILS_STAGE2), $(RFAIL_RS))

TEST_RFAIL_EXES_STAGE0 := \
  $(subst $(S)src/,,$(TEST_RFAIL_CRATES_STAGE0:.rc=.stage0$(X))) \
  $(subst $(S)src/,,$(TEST_RFAIL_SOURCES_STAGE0:.rs=.stage0$(X)))
TEST_RFAIL_EXES_STAGE1 := \
  $(subst $(S)src/,,$(TEST_RFAIL_CRATES_STAGE1:.rc=.stage1$(X))) \
  $(subst $(S)src/,,$(TEST_RFAIL_SOURCES_STAGE1:.rs=.stage1$(X)))
TEST_RFAIL_EXES_STAGE2 := \
  $(subst $(S)src/,,$(TEST_RFAIL_CRATES_STAGE2:.rc=.stage2$(X))) \
  $(subst $(S)src/,,$(TEST_RFAIL_SOURCES_STAGE2:.rs=.stage2$(X)))

TEST_RFAIL_OUTS_STAGE0 := \
  $(TEST_RFAIL_EXES_STAGE0:.stage0$(X)=.stage0.out)
TEST_RFAIL_OUTS_STAGE1 := \
  $(TEST_RFAIL_EXES_STAGE1:.stage1$(X)=.stage1.out)
TEST_RFAIL_OUTS_STAGE2 := \
  $(TEST_RFAIL_EXES_STAGE2:.stage2$(X)=.stage2.out)


TEST_CFAIL_CRATES_STAGE0 := $(filter-out $(TEST_XFAILS_STAGE0), $(CFAIL_RC))
TEST_CFAIL_CRATES_STAGE1 := $(filter-out $(TEST_XFAILS_STAGE1), $(CFAIL_RC))
TEST_CFAIL_CRATES_STAGE2 := $(filter-out $(TEST_XFAILS_STAGE2), $(CFAIL_RC))
TEST_CFAIL_SOURCES_STAGE0 := $(filter-out $(TEST_XFAILS_STAGE0), $(CFAIL_RS))
TEST_CFAIL_SOURCES_STAGE1 := $(filter-out $(TEST_XFAILS_STAGE1), $(CFAIL_RS))
TEST_CFAIL_SOURCES_STAGE2 := $(filter-out $(TEST_XFAILS_STAGE2), $(CFAIL_RS))

TEST_CFAIL_OUTS_STAGE0 := \
  $(subst $(S)src/,,$(TEST_CFAIL_CRATES_STAGE0:.rc=.stage0.out)) \
  $(subst $(S)src/,,$(TEST_CFAIL_SOURCES_STAGE0:.rs=.stage0.out))
TEST_CFAIL_OUTS_STAGE1 := \
  $(subst $(S)src/,,$(TEST_CFAIL_CRATES_STAGE1:.rc=.stage1.out)) \
  $(subst $(S)src/,,$(TEST_CFAIL_SOURCES_STAGE1:.rs=.stage1.out))
TEST_CFAIL_OUTS_STAGE2 := \
  $(subst $(S)src/,,$(TEST_CFAIL_CRATES_STAGE2:.rc=.stage2.out)) \
  $(subst $(S)src/,,$(TEST_CFAIL_SOURCES_STAGE2:.rs=.stage2.out))


ALL_TEST_CRATES := $(TEST_CFAIL_CRATES_STAGE0) \
                   $(TEST_RFAIL_CRATES_STAGE0) \
                   $(TEST_RPASS_CRATES_STAGE0) \
                   $(TEST_CFAIL_CRATES_STAGE1) \
                   $(TEST_RFAIL_CRATES_STAGE1) \
                   $(TEST_RPASS_CRATES_STAGE1) \
                   $(TEST_CFAIL_CRATES_STAGE2) \
                   $(TEST_RFAIL_CRATES_STAGE2) \
                   $(TEST_RPASS_CRATES_STAGE2)

ALL_TEST_SOURCES := $(TEST_CFAIL_SOURCES_STAGE0) \
                    $(TEST_RFAIL_SOURCES_STAGE0) \
                    $(TEST_RPASS_SOURCES_STAGE0) \
                    $(TEST_CFAIL_SOURCES_STAGE1) \
                    $(TEST_RFAIL_SOURCES_STAGE1) \
                    $(TEST_RPASS_SOURCES_STAGE1) \
                    $(TEST_CFAIL_SOURCES_STAGE2) \
                    $(TEST_RFAIL_SOURCES_STAGE2) \
                    $(TEST_RPASS_SOURCES_STAGE2)

check-nocompile: $(TEST_CFAIL_OUTS_STAGE0) \
                 $(TEST_CFAIL_OUTS_STAGE1) \
                 $(TEST_CFAIL_OUTS_STAGE2)

check-stage0: tidy \
       $(TEST_RPASS_EXES_STAGE0) $(TEST_RFAIL_EXES_STAGE0) \
       $(TEST_RPASS_OUTS_STAGE0) $(TEST_RFAIL_OUTS_STAGE0) \
       $(TEST_CFAIL_OUTS_STAGE0) \


check-stage1: tidy \
       $(TEST_RPASS_EXES_STAGE1) $(TEST_RFAIL_EXES_STAGE1) \
       $(TEST_RPASS_OUTS_STAGE1) $(TEST_RFAIL_OUTS_STAGE1) \
       $(TEST_CFAIL_OUTS_STAGE1) \


check-stage2: tidy \
       $(TEST_RPASS_EXES_STAGE2) $(TEST_RFAIL_EXES_STAGE2) \
       $(TEST_RPASS_OUTS_STAGE2) $(TEST_RFAIL_OUTS_STAGE2) \
       $(TEST_CFAIL_OUTS_STAGE2) \


check: tidy \
       $(TEST_RPASS_EXES_STAGE2) $(TEST_RFAIL_EXES_STAGE2) \
       $(TEST_RPASS_OUTS_STAGE2) $(TEST_RFAIL_OUTS_STAGE2) \
       $(TEST_CFAIL_OUTS_STAGE2)

full-check: tidy \
       $(TEST_RPASS_EXES_STAGE0) $(TEST_RFAIL_EXES_STAGE0) \
       $(TEST_RPASS_OUTS_STAGE0) $(TEST_RFAIL_OUTS_STAGE0) \
       $(TEST_CFAIL_OUTS_STAGE0) \
       $(TEST_RPASS_EXES_STAGE1) $(TEST_RFAIL_EXES_STAGE1) \
       $(TEST_RPASS_OUTS_STAGE1) $(TEST_RFAIL_OUTS_STAGE1) \
       $(TEST_CFAIL_OUTS_STAGE1) \
       $(TEST_RPASS_EXES_STAGE2) $(TEST_RFAIL_EXES_STAGE2) \
       $(TEST_RPASS_OUTS_STAGE2) $(TEST_RFAIL_OUTS_STAGE2) \
       $(TEST_CFAIL_OUTS_STAGE2)

compile-check: tidy \
       $(TEST_RPASS_EXES_STAGE0) $(TEST_RFAIL_EXES_STAGE0) \
       $(TEST_RPASS_EXES_STAGE1) $(TEST_RFAIL_EXES_STAGE1) \
       $(TEST_RPASS_EXES_STAGE2) $(TEST_RFAIL_EXES_STAGE2)


######################################################################
# Testing rules
######################################################################

tidy:
	@$(call E, check: formatting)
	$(Q)echo \
      $(filter-out $(GENERATED) $(addprefix $(S)src/, $(GENERATED)) \
        $(addprefix $(S)src/, $(RUSTLLVM_LIB_CS) $(RUSTLLVM_OBJS_CS) \
          $(RUSTLLVM_HDR) $(PKG_3RDPARTY)) \
        $(S)src/etc/%, $(PKG_FILES)) \
    | xargs -n 10 python $(S)src/etc/tidy.py

%.stage0$(X): %.rs $(SREQ0)
	@$(call E, compile_and_link: $@)
	$(STAGE0) -o $@ $<

%.stage0$(X): %.rc $(SREQ0)
	@$(call E, compile_and_link: $@)
	$(STAGE0) -o $@ $<

%.stage1$(X): %.rs $(SREQ1)
	@$(call E, compile_and_link: $@)
	$(STAGE1) -o $@ $<

%.stage1$(X): %.rc $(SREQ1)
	@$(call E, compile_and_link: $@)
	$(STAGE1) -o $@ $<

%.stage2$(X): %.rs $(SREQ2)
	@$(call E, compile_and_link: $@)
	$(STAGE2) -o $@ $<

%.stage2$(X): %.rc $(SREQ2)
	@$(call E, compile_and_link: $@)
	$(STAGE2) -o $@ $<

# Cancel the implicit .out rule in GNU make.
%.out: %

%.out: %.out.tmp
	$(Q)mv $< $@

test/run-pass/%.out.tmp: test/run-pass/%$(X) rt/$(CFG_RUNTIME)
	$(Q)rm -f $<.tmp
	@$(call E, run: $@)
	$(Q)$(call CFG_RUN_TEST, $<) > $@

test/bench/shootout/%.out.tmp: test/bench/shootout/%$(X) \
                               rt/$(CFG_RUNTIME)
	$(Q)rm -f $<.tmp
	@$(call E, run: $@)
	$(Q)$(call CFG_RUN_TEST, $<) > $@

test/bench/99-bottles/%.out.tmp: test/bench/99-bottles/%$(X) \
                                 rt/$(CFG_RUNTIME)
	$(Q)rm -f $<.tmp
	@$(call E, run: $@)
	$(Q)$(call CFG_RUN_TEST, $<) > $@

test/run-fail/%.out.tmp: test/run-fail/%$(X) \
                         rt/$(CFG_RUNTIME)
	$(Q)rm -f $<.tmp
	@$(call E, run-fail: $@)
	$(Q)grep -q error-pattern $(S)src/test/run-fail/$(basename $*).rs
	$(Q)rm -f $@
	$(Q)$(call CFG_RUN_TEST, $<) >$@ 2>&1 ; X=$$? ; \
      if [ $$X -eq 0 ] ; then exit 1 ; else exit 0 ; fi
	$(Q)grep --text --quiet \
      "$$(grep error-pattern $(S)src/test/run-fail/$(basename $*).rs \
        | cut -d : -f 2- | tr -d '\n\r')" $@

test/compile-fail/%.stage0.out.tmp: test/compile-fail/%.rs $(SREQ0)
	@$(call E, compile-fail [stage0]: $@)
	$(Q)grep -q error-pattern $<
	$(Q)rm -f $@
	$(STAGE0) -c -o $(@:.o=$(X)) $< >$@ 2>&1; test $$? -ne 0
	$(Q)grep --text --quiet \
      "$$(grep error-pattern $< | cut -d : -f 2- | tr -d '\n\r')" $@

test/compile-fail/%.stage1.out.tmp: test/compile-fail/%.rs $(SREQ1)
	@$(call E, compile-fail [stage1]: $@)
	$(Q)grep -q error-pattern $<
	$(Q)rm -f $@
	$(STAGE1) -c -o $(@:.o=$(X)) $< >$@ 2>&1; test $$? -ne 0
	$(Q)grep --text --quiet \
      "$$(grep error-pattern $< | cut -d : -f 2- | tr -d '\n\r')" $@

test/compile-fail/%.stage2.out.tmp: test/compile-fail/%.rs $(SREQ2)
	@$(call E, compile-fail [stage2]: $@)
	$(Q)grep -q error-pattern $<
	$(Q)rm -f $@
	$(STAGE2) -c -o $(@:.o=$(X)) $< >$@ 2>&1; test $$? -ne 0
	$(Q)grep --text --quiet \
      "$$(grep error-pattern $< | cut -d : -f 2- | tr -d '\n\r')" $@

test/compile-fail/%.stage0.out.tmp: test/compile-fail/%.rc $(SREQ0)
	@$(call E, compile-fail [stage0]: $@)
	$(Q)grep -q error-pattern $<
	$(Q)rm -f $@
	$(STAGE0) -c -o $(@:.o=$(X)) $< >$@ 2>&1; test $$? -ne 0
	$(Q)grep --text --quiet \
      "$$(grep error-pattern $< | cut -d : -f 2- | tr -d '\n\r')" $@

test/compile-fail/%.stage1.out.tmp: test/compile-fail/%.rc $(SREQ1)
	@$(call E, compile-fail [stage1]: $@)
	$(Q)grep -q error-pattern $<
	$(Q)rm -f $@
	$(STAGE1) -c -o $(@:.o=$(X)) $< >$@ 2>&1; test $$? -ne 0
	$(Q)grep --text --quiet \
      "$$(grep error-pattern $< | cut -d : -f 2- | tr -d '\n\r')" $@

test/compile-fail/%.stage2.out.tmp: test/compile-fail/%.rc $(SREQ2)
	@$(call E, compile-fail [stage2]: $@)
	$(Q)grep -q error-pattern $<
	$(Q)rm -f $@
	$(STAGE2) -c -o $(@:.o=$(X)) $< >$@ 2>&1; test $$? -ne 0
	$(Q)grep --text --quiet \
      "$$(grep error-pattern $< | cut -d : -f 2- | tr -d '\n\r')" $@
