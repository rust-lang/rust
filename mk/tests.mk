######################################################################
# Testing variables
######################################################################

ALL_TEST_INPUTS = $(wildcard $(S)src/test/*/*.rs   \
                              $(S)src/test/*/*/*.rs \
                              $(S)src/test/*/*.rc)

TEST_XFAILS_STAGE0 = $(shell grep -l xfail-stage0 $(ALL_TEST_INPUTS))
TEST_XFAILS_STAGE1 = $(shell grep -l xfail-stage1 $(ALL_TEST_INPUTS))
TEST_XFAILS_STAGE2 = $(shell grep -l xfail-stage2 $(ALL_TEST_INPUTS))

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

BENCH_RS = $(wildcard $(S)src/test/bench/shootout/*.rs) \
            $(wildcard $(S)src/test/bench/99-bottles/*.rs)
RPASS_RC = $(wildcard $(S)src/test/run-pass/*.rc)
RPASS_RS = $(wildcard $(S)src/test/run-pass/*.rs) $(BENCH_RS)
RFAIL_RC = $(wildcard $(S)src/test/run-fail/*.rc)
RFAIL_RS = $(wildcard $(S)src/test/run-fail/*.rs)
CFAIL_RC = $(wildcard $(S)src/test/compile-fail/*.rc)
CFAIL_RS = $(wildcard $(S)src/test/compile-fail/*.rs)

ifdef CHECK_XFAILS
TEST_RPASS_CRATES_STAGE0 = $(filter $(TEST_XFAILS_STAGE0), $(RPASS_RC))
TEST_RPASS_CRATES_STAGE1 = $(filter $(TEST_XFAILS_STAGE1), $(RPASS_RC))
TEST_RPASS_CRATES_STAGE2 = $(filter $(TEST_XFAILS_STAGE2), $(RPASS_RC))
TEST_RPASS_SOURCES_STAGE0 = $(filter $(TEST_XFAILS_STAGE0), $(RPASS_RS))
TEST_RPASS_SOURCES_STAGE1 = $(filter $(TEST_XFAILS_STAGE1), $(RPASS_RS))
TEST_RPASS_SOURCES_STAGE2 = $(filter $(TEST_XFAILS_STAGE2), $(RPASS_RS))
else
TEST_RPASS_CRATES_STAGE0 = $(filter-out $(TEST_XFAILS_STAGE0), $(RPASS_RC))
TEST_RPASS_CRATES_STAGE1 = $(filter-out $(TEST_XFAILS_STAGE1), $(RPASS_RC))
TEST_RPASS_CRATES_STAGE1 = $(filter-out $(TEST_XFAILS_STAGE2), $(RPASS_RC))
TEST_RPASS_SOURCES_STAGE0 = $(filter-out $(TEST_XFAILS_STAGE0), $(RPASS_RS))
TEST_RPASS_SOURCES_STAGE1 = $(filter-out $(TEST_XFAILS_STAGE1), $(RPASS_RS))
TEST_RPASS_SOURCES_STAGE2 = $(filter-out $(TEST_XFAILS_STAGE2), $(RPASS_RS))
endif

TEST_RPASS_EXES_STAGE0 = \
  $(subst $(S)src/,,$(TEST_RPASS_CRATES_STAGE0:.rc=.stage0$(X))) \
  $(subst $(S)src/,,$(TEST_RPASS_SOURCES_STAGE0:.rs=.stage0$(X)))
TEST_RPASS_EXES_STAGE1 = \
  $(subst $(S)src/,,$(TEST_RPASS_CRATES_STAGE1:.rc=.stage1$(X))) \
  $(subst $(S)src/,,$(TEST_RPASS_SOURCES_STAGE1:.rs=.stage1$(X)))
TEST_RPASS_EXES_STAGE2 = \
  $(subst $(S)src/,,$(TEST_RPASS_CRATES_STAGE1:.rc=.stage2$(X))) \
  $(subst $(S)src/,,$(TEST_RPASS_SOURCES_STAGE1:.rs=.stage2$(X)))

TEST_RPASS_OUTS_STAGE0 = \
  $(TEST_RPASS_EXES_STAGE0:.stage0$(X)=.stage0.out)
TEST_RPASS_OUTS_STAGE1 = \
  $(TEST_RPASS_EXES_STAGE1:.stage1$(X)=.stage1.out)
TEST_RPASS_OUTS_STAGE2 = \
  $(TEST_RPASS_EXES_STAGE2:.stage2$(X)=.stage2.out)

TEST_RPASS_TMPS_STAGE0 = \
  $(TEST_RPASS_EXES_STAGE0:.stage0$(X)=.stage0$(X).tmp)
TEST_RPASS_TMPS_STAGE1 = \
  $(TEST_RPASS_EXES_STAGE1:.stage1$(X)=.stage1$(X).tmp)
TEST_RPASS_TMPS_STAGE2 = \
  $(TEST_RPASS_EXES_STAGE2:.stage2$(X)=.stage2$(X).tmp)


TEST_RFAIL_CRATES_STAGE0 = $(filter-out $(TEST_XFAILS_STAGE0), $(RFAIL_RC))
TEST_RFAIL_CRATES_STAGE1 = $(filter-out $(TEST_XFAILS_STAGE1), $(RFAIL_RC))
TEST_RFAIL_CRATES_STAGE2 = $(filter-out $(TEST_XFAILS_STAGE2), $(RFAIL_RC))
TEST_RFAIL_SOURCES_STAGE0 = $(filter-out $(TEST_XFAILS_STAGE0), $(RFAIL_RS))
TEST_RFAIL_SOURCES_STAGE1 = $(filter-out $(TEST_XFAILS_STAGE1), $(RFAIL_RS))
TEST_RFAIL_SOURCES_STAGE2 = $(filter-out $(TEST_XFAILS_STAGE2), $(RFAIL_RS))

TEST_RFAIL_EXES_STAGE0 = \
  $(subst $(S)src/,,$(TEST_RFAIL_CRATES_STAGE0:.rc=.stage0$(X))) \
  $(subst $(S)src/,,$(TEST_RFAIL_SOURCES_STAGE0:.rs=.stage0$(X)))
TEST_RFAIL_EXES_STAGE1 = \
  $(subst $(S)src/,,$(TEST_RFAIL_CRATES_STAGE1:.rc=.stage1$(X))) \
  $(subst $(S)src/,,$(TEST_RFAIL_SOURCES_STAGE1:.rs=.stage1$(X)))
TEST_RFAIL_EXES_STAGE2 = \
  $(subst $(S)src/,,$(TEST_RFAIL_CRATES_STAGE2:.rc=.stage2$(X))) \
  $(subst $(S)src/,,$(TEST_RFAIL_SOURCES_STAGE2:.rs=.stage2$(X)))

TEST_RFAIL_OUTS_STAGE0 = \
  $(TEST_RFAIL_EXES_STAGE0:.stage0$(X)=.stage0.out)
TEST_RFAIL_OUTS_STAGE1 = \
  $(TEST_RFAIL_EXES_STAGE1:.stage1$(X)=.stage1.out)
TEST_RFAIL_OUTS_STAGE2 = \
  $(TEST_RFAIL_EXES_STAGE2:.stage2$(X)=.stage2.out)


TEST_CFAIL_CRATES_STAGE0 = $(filter-out $(TEST_XFAILS_STAGE0), $(CFAIL_RC))
TEST_CFAIL_CRATES_STAGE1 = $(filter-out $(TEST_XFAILS_STAGE1), $(CFAIL_RC))
TEST_CFAIL_CRATES_STAGE2 = $(filter-out $(TEST_XFAILS_STAGE2), $(CFAIL_RC))
TEST_CFAIL_SOURCES_STAGE0 = $(filter-out $(TEST_XFAILS_STAGE0), $(CFAIL_RS))
TEST_CFAIL_SOURCES_STAGE1 = $(filter-out $(TEST_XFAILS_STAGE1), $(CFAIL_RS))
TEST_CFAIL_SOURCES_STAGE2 = $(filter-out $(TEST_XFAILS_STAGE2), $(CFAIL_RS))

TEST_CFAIL_OUTS_STAGE0 = \
  $(subst $(S)src/,,$(TEST_CFAIL_CRATES_STAGE0:.rc=.stage0.out)) \
  $(subst $(S)src/,,$(TEST_CFAIL_SOURCES_STAGE0:.rs=.stage0.out))
TEST_CFAIL_OUTS_STAGE1 = \
  $(subst $(S)src/,,$(TEST_CFAIL_CRATES_STAGE1:.rc=.stage1.out)) \
  $(subst $(S)src/,,$(TEST_CFAIL_SOURCES_STAGE1:.rs=.stage1.out))
TEST_CFAIL_OUTS_STAGE2 = \
  $(subst $(S)src/,,$(TEST_CFAIL_CRATES_STAGE2:.rc=.stage2.out)) \
  $(subst $(S)src/,,$(TEST_CFAIL_SOURCES_STAGE2:.rs=.stage2.out))


ALL_TEST_CRATES =  $(TEST_CFAIL_CRATES_STAGE0) \
                   $(TEST_RFAIL_CRATES_STAGE0) \
                   $(TEST_RPASS_CRATES_STAGE0) \
                   $(TEST_CFAIL_CRATES_STAGE1) \
                   $(TEST_RFAIL_CRATES_STAGE1) \
                   $(TEST_RPASS_CRATES_STAGE1) \
                   $(TEST_CFAIL_CRATES_STAGE2) \
                   $(TEST_RFAIL_CRATES_STAGE2) \
                   $(TEST_RPASS_CRATES_STAGE2)

ALL_TEST_SOURCES =  $(TEST_CFAIL_SOURCES_STAGE0) \
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

%.stage0$(X): %.stage0.o  $(SREQ0)
	@$(call E, link [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage1/glue.o -o $@ $< \
      -Lstage1 -Lrt rt/main.o -lrustrt -lstd -lm
	@# dsymutil sometimes fails or prints a warning, but the
	@# program still runs.  Since it simplifies debugging other
	@# programs, I\'ll live with the noise.
	-$(Q)$(CFG_DSYMUTIL) $@

%.stage1$(X): %.stage1.o $(SREQ1)
	@$(call E, link [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage2/glue.o -o $@ $< \
      -Lstage2 -Lrt rt/main.o -lrustrt -lstd -lm
	@# dsymutil sometimes fails or prints a warning, but the
	@# program still runs.  Since it simplifies debugging other
	@# programs, I\'ll live with the noise.
	-$(Q)$(CFG_DSYMUTIL) $@

%.stage2$(X): %.stage2.o $(SREQ2)
	@$(call E, link [gcc]: $@)
	$(Q)gcc $(CFG_GCCISH_CFLAGS) stage3/glue.o -o $@ $< \
      -Lstage3 -Lrt rt/main.o -lrustrt -lstd -lm
	@# dsymutil sometimes fails or prints a warning, but the
	@# program still runs.  Since it simplifies debugging other
	@# programs, I\'ll live with the noise.
	-$(Q)$(CFG_DSYMUTIL) $@


%.stage0.o: %.rc $(SREQ0)
	@$(call E, compile [stage0]: $@)
	$(STAGE0) -c -o $@ $<

%.stage0.o: %.rs $(SREQ0)
	@$(call E, compile [stage0]: $@)
	$(STAGE0) -c -o $@ $<


%.stage1.o: %.rc $(SREQ1)
	@$(call E, compile [stage1]: $@)
	$(STAGE1) -c -o $@ $<

%.stage1.o: %.rs $(SREQ1)
	@$(call E, compile [stage1]: $@)
	$(STAGE1) -c -o $@ $<


%.stage2.o: %.rc $(SREQ2)
	@$(call E, compile [stage2]: $@)
	$(STAGE2) -c -o $@ $<

%.stage2.o: %.rs $(SREQ2)
	@$(call E, compile [stage2]: $@)
	$(STAGE2) -c -o $@ $<

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
