######################################################################
# Testing variables
######################################################################

ALL_TEST_INPUTS = $(wildcard $(S)src/test/*/*.rs   \
                              $(S)src/test/*/*/*.rs \
                              $(S)src/test/*/*.rc)

RPASS_RC := $(wildcard $(S)src/test/run-pass/*.rc)
RPASS_RS := $(wildcard $(S)src/test/run-pass/*.rs)
RFAIL_RC := $(wildcard $(S)src/test/run-fail/*.rc)
RFAIL_RS := $(wildcard $(S)src/test/run-fail/*.rs)
CFAIL_RC := $(wildcard $(S)src/test/compile-fail/*.rc)
CFAIL_RS := $(wildcard $(S)src/test/compile-fail/*.rs)
BENCH_RS := $(wildcard $(S)src/test/bench/*.rs)
PRETTY_RS := $(wildcard $(S)src/test/pretty/*.rs)

# perf tests are the same as bench tests only they run under
# a performance monitor.
PERF_RS := $(wildcard $(S)src/test/bench/*.rs)

RPASS_TESTS := $(RPASS_RC) $(RPASS_RS)
RFAIL_TESTS := $(RFAIL_RC) $(RFAIL_RS)
CFAIL_TESTS := $(CFAIL_RC) $(CFAIL_RS)
BENCH_TESTS := $(BENCH_RS)
PERF_TESTS := $(PERF_RS)
PRETTY_TESTS := $(PRETTY_RS)

FT := run_pass_stage2
FT_LIB := $(call CFG_LIB_NAME,$(FT))
FT_DRIVER := $(FT)_driver
GENERATED += test/$(FT).rc test/$(FT_DRIVER).rs

# The arguments to all test runners
ifdef TESTNAME
  TESTARGS += $(TESTNAME)
endif

ifdef CHECK_XFAILS
  TESTARGS += --ignored
endif

# Arguments to the cfail/rfail/rpass/bench tests
ifdef CFG_VALGRIND
  CTEST_RUNTOOL = --runtool "$(CFG_VALGRIND)"
endif

# Arguments to the perf tests
ifdef CFG_PERF_TOOL
  CTEST_PERF_RUNTOOL = --runtool "$(CFG_PERF_TOOL)"
endif

CTEST_TESTARGS := $(TESTARGS)

ifdef VERBOSE
  CTEST_TESTARGS += --verbose
endif

# The test runner that runs the cfail/rfail/rpass and bench tests
COMPILETEST_CRATE := $(S)src/test/compiletest/compiletest.rc
COMPILETEST_INPUTS := $(wildcard $(S)src/test/compiletest/*rs)

# The standard library test crate
STDTEST_CRATE := $(S)src/test/stdtest/stdtest.rc
STDTEST_INPUTS := $(wildcard $(S)src/test/stdtest/*rs)

# Run the compiletest runner itself under valgrind
ifdef CTEST_VALGRIND
  CFG_RUN_CTEST=$(call CFG_RUN_TEST,$(2))
else
  CFG_RUN_CTEST=$(call CFG_RUN,stage$(1)/lib,$(2))
endif

######################################################################
# Main test targets
######################################################################

check: tidy check-stage2 \

check-full: tidy check-stage1 check-stage2 check-stage3 \

check-fast: tidy \
	check-stage2-rustc check-stage2-std \
	test/$(FT_DRIVER).out \

tidy:
	@$(call E, check: formatting)
	$(Q)echo \
	  $(filter-out $(GENERATED) $(addprefix $(S)src/, $(GENERATED)) \
	    $(addprefix $(S)src/, $(RUSTLLVM_LIB_CS) $(RUSTLLVM_OBJS_CS) \
	      $(RUSTLLVM_HDR) $(PKG_3RDPARTY)) \
	    $(S)src/etc/%, $(PKG_FILES)) \
	  | xargs -n 10 python $(S)src/etc/tidy.py

# Cancel the implicit .out rule in GNU make
%.out: %

%.out: %.out.tmp
	$(Q)mv $< $@


######################################################################
# Rules for the test runners
######################################################################

# StageN template: to stay consistent with stageN.mk, arge 2 is the
# stage being tested, arg 1 is stage N-1

define TEST_STAGEN

# FIXME: These rules are complicated by a scheme to produce .out files
# for each test, with the idea that if the targets produce actual
# output, subsequent "make check's" will just exit successfully
# without re-running the tests. Unfortunately this scheme is currently
# broken and the tests rerun with every invocation of "make check"
# anyway. Nobody seems to have noticed, so it would be simpler to just
# remove all the code here involved with producing .out files.

check-stage$(2): tidy \
	check-stage$(2)-rustc \
	check-stage$(2)-std \
	check-stage$(2)-rpass \
	check-stage$(2)-rfail \
	check-stage$(2)-cfail \
	check-stage$(2)-bench \
	check-stage$(2)-pretty \


# Rules for the standard library test runner

check-stage$(2)-std: test/stdtest.stage$(2).out \

test/stdtest.stage$(2)$$(X): $$(STDTEST_CRATE) $$(STDTEST_INPUTS) \
                             $$(SREQ$(2))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2)) -o $$@ $$< --test

test/stdtest.stage$(2).out.tmp: test/stdtest.stage$(2)$$(X)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_TEST,$$<) $$(TESTARGS)
	$$(Q)touch $$@


# Rules for the rustc test runner

check-stage$(2)-rustc: test/rustctest.stage$(2).out \

test/rustctest.stage$(2)$$(X): $$(COMPILER_CRATE) $$(COMPILER_INPUTS) \
                           stage$(2)/$$(CFG_RUNTIME) \
                           $$(call CFG_STDLIB_DEFAULT,stage$(1),stage$(2)) \
                           stage$(2)/$$(CFG_RUSTLLVM) \
                           $$(SREQ$(1))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)) -o $$@ $$< --test

test/rustctest.stage$(2).out.tmp: test/rustctest.stage$(2)$$(X)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN,stage$(2),$$(CFG_VALGRIND) $$<) \
	  $$(TESTARGS)
	$$(Q)touch $$@


# Rules for the cfail/rfail/rpass/bench/perf test runner

check-stage$(2)-cfail: test/compile-fail.stage$(2).out \

check-stage$(2)-rfail: test/run-fail.stage$(2).out \

check-stage$(2)-rpass: test/run-pass.stage$(2).out \

check-stage$(2)-bench: test/bench.stage$(2).out \

check-stage$(2)-perf: test/perf.stage$(2).out \

check-stage$(2)-pretty-rpass: test/pretty-rpass.stage$(2).out \

check-stage$(2)-pretty-rfail: test/pretty-rfail.stage$(2).out \

check-stage$(2)-pretty-bench: test/pretty-bench.stage$(2).out \

check-stage$(2)-pretty-pretty: test/pretty-pretty.stage$(2).out \

check-stage$(2)-pretty: check-stage$(2)-pretty-rpass \
                        check-stage$(2)-pretty-rfail \
                        check-stage$(2)-pretty-bench \
                        check-stage$(2)-pretty-pretty \

CTEST_COMMON_ARGS$(2) := --compile-lib-path stage$(2) \
                         --run-lib-path stage$(2)/lib \
                         --rustc-path stage$(2)/rustc$$(X) \
                         --stage-id stage$(2) \
                         --rustcflags "$$(CFG_RUSTC_FLAGS)" \
                         $$(CTEST_TESTARGS) \

CFAIL_ARGS$(2) := $$(CTEST_COMMON_ARGS$(2)) \
                  --src-base $$(S)src/test/compile-fail/ \
                  --build-base test/compile-fail/ \
                  --mode compile-fail \

RFAIL_ARGS$(2) := $$(CTEST_COMMON_ARGS$(2)) \
                  --src-base $$(S)src/test/run-fail/ \
                  --build-base test/run-fail/ \
                  --mode run-fail \
                  $$(CTEST_RUNTOOL) \

RPASS_ARGS$(2) := $$(CTEST_COMMON_ARGS$(2)) \
                  --src-base $(S)src/test/run-pass/ \
                  --build-base test/run-pass/ \
                  --mode run-pass \
                  $$(CTEST_RUNTOOL) \

BENCH_ARGS$(2) := $$(CTEST_COMMON_ARGS$(2)) \
                  --src-base $(S)src/test/bench/ \
                  --build-base test/bench/ \
                  --mode run-pass \
                  $$(CTEST_RUNTOOL) \

PERF_ARGS$(2) := $$(CTEST_COMMON_ARGS$(2)) \
                  --src-base $(S)src/test/bench/ \
                  --build-base test/perf/ \
                  --mode run-pass \
                  $$(CTEST_PERF_RUNTOOL) \

PRETTY_RPASS_ARGS$(2) := $$(CTEST_COMMON_ARGS$(2)) \
                         --src-base $$(S)src/test/run-pass/ \
                         --build-base test/run-pass/ \
                         --mode pretty \

PRETTY_RFAIL_ARGS$(2) := $$(CTEST_COMMON_ARGS$(2)) \
                         --src-base $$(S)src/test/run-fail/ \
                         --build-base test/run-fail/ \
                         --mode pretty \

PRETTY_BENCH_ARGS$(2) := $$(CTEST_COMMON_ARGS$(2)) \
                         --src-base $$(S)src/test/bench/ \
                         --build-base test/bench/ \
                         --mode pretty \

PRETTY_PRETTY_ARGS$(2) := $$(CTEST_COMMON_ARGS$(2)) \
                          --src-base $$(S)src/test/pretty/ \
                          --build-base test/pretty/ \
                          --mode pretty \

test/compiletest.stage$(2)$$(X): $$(COMPILETEST_CRATE) \
                                 $$(COMPILETEST_INPUTS) \
                                 $$(SREQ$(2))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(2)) -o $$@ $$<

test/compile-fail.stage$(2).out.tmp: test/compiletest.stage$(2)$$(X) \
                                   $$(CFAIL_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(2),$$<) $$(CFAIL_ARGS$(2))
	$$(Q)touch $$@

test/run-fail.stage$(2).out.tmp: test/compiletest.stage$(2)$$(X) \
                               $$(RFAIL_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(2),$$<) $$(RFAIL_ARGS$(2))
	$$(Q)touch $$@

test/run-pass.stage$(2).out.tmp: test/compiletest.stage$(2)$$(X) \
                               $$(RPASS_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(2),$$<) $$(RPASS_ARGS$(2))
	$$(Q)touch $$@

test/bench.stage$(2).out.tmp: test/compiletest.stage$(2)$$(X) \
                            $$(BENCH_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(2),$$<) $$(BENCH_ARGS$(2))
	$$(Q)touch $$@

test/perf.stage$(2).out.tmp: test/compiletest.stage$(2)$$(X) \
                            $$(BENCH_TESTS)
	@$$(call E, perf: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(2),$$<) $$(PERF_ARGS$(2))
	$$(Q)touch $$@

test/pretty-rpass.stage$(2).out.tmp: test/compiletest.stage$(2)$$(X) \
                                     $$(RPASS_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(2),$$<) $$(PRETTY_RPASS_ARGS$(2))
	$$(Q)touch $$@

test/pretty-rfail.stage$(2).out.tmp: test/compiletest.stage$(2)$$(X) \
                                     $$(RFAIL_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(2),$$<) $$(PRETTY_RFAIL_ARGS$(2))
	$$(Q)touch $$@

test/pretty-bench.stage$(2).out.tmp: test/compiletest.stage$(2)$$(X) \
                                     $$(BENCH_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(2),$$<) $$(PRETTY_BENCH_ARGS$(2))
	$$(Q)touch $$@

test/pretty-pretty.stage$(2).out.tmp: test/compiletest.stage$(2)$$(X) \
                                     $$(PRETTY_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(2),$$<) $$(PRETTY_PRETTY_ARGS$(2))
	$$(Q)touch $$@

endef

# Instantiate the template for stage 0, 1, 2, 3

$(eval $(call TEST_STAGEN,0,0))
$(eval $(call TEST_STAGEN,0,1))
$(eval $(call TEST_STAGEN,1,2))
$(eval $(call TEST_STAGEN,2,3))


######################################################################
# Fast-test rules
######################################################################

test/$(FT).rc test/$(FT_DRIVER).rs: $(TEST_RPASS_SOURCES_STAGE2) \
    $(S)src/etc/combine-tests.py
	@$(call E, check: building combined stage2 test runner)
	$(Q)$(S)src/etc/combine-tests.py

stage2/lib/$(FT_LIB): test/$(FT).rc $(SREQ2)
	@$(call E, compile_and_link: $@)
	$(STAGE2) --lib -o $@ $<

test/$(FT_DRIVER)$(X): test/$(FT_DRIVER).rs stage2/lib/$(FT_LIB) $(SREQ2)
	@$(call E, compile_and_link: $@)
	$(STAGE2) -o $@ $<

test/$(FT_DRIVER).out: test/$(FT_DRIVER)$(X) $(SREQ2)
	$(Q)$(call CFG_RUN_TEST, $<)
