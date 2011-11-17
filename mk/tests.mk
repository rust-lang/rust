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

# The standard library test crate
STDTEST_CRATE := $(S)src/test/stdtest/stdtest.rc
STDTEST_INPUTS := $(wildcard $(S)src/test/stdtest/*rs)

# Run the compiletest runner itself under valgrind
ifdef CTEST_VALGRIND
  CFG_RUN_CTEST=$(call CFG_RUN_TEST,$(2))
else
  CFG_RUN_CTEST=$(call CFG_RUN,$(TARGET_HOST_LIB$(1)),$(2))
endif

######################################################################
# Main test targets
######################################################################

check: tidy check-stage2 \

check-full: tidy check-stage1 check-stage2 check-stage3 \

check-fast: tidy \
	check-stage2-rustc check-stage2-std \
	test/$(FT_DRIVER).out

# Run the tidy script in multiple parts to avoid huge 'echo' commands
tidy:
	@$(call E, check: formatting)
	$(Q)echo \
	    $(addprefix $(S)src/, $(RUSTLLVM_LIB_CS) $(RUSTLLVM_OBJS_CS) \
	      $(RUSTLLVM_HDR) \
              $(RUNTIME_CS) $(RUNTIME_HDR) $(RUNTIME_S)) \
            $(wildcard $(S)src/etc/*.py)  \
            $(COMPILER_CRATE) \
            $(COMPILER_INPUTS) \
            $(STDLIB_CRATE) \
            $(STDLIB_INPUTS) \
            $(COMPILETEST_CRATE) \
            $(COMPILETEST_INPUTS) \
	  | xargs -n 10 python $(S)src/etc/tidy.py
	$(Q)echo \
            $(ALL_TEST_INPUTS) \
	  | xargs -n 10 python $(S)src/etc/tidy.py


######################################################################
# Rules for the test runners
######################################################################

define TEST_STAGEN

# All the per-stage build rules you might want to call from the
# command line.
#
# $(1) is the stage number
# $(2) is the target triple

check-stage$(1)-$(2): tidy \
	check-stage$(1)-$(2)-rustc \
	check-stage$(1)-$(2)-std \
	check-stage$(1)-$(2)-rpass \
	check-stage$(1)-$(2)-rfail \
	check-stage$(1)-$(2)-cfail \
	check-stage$(1)-$(2)-bench \
	check-stage$(1)-$(2)-pretty

check-stage$(1)-$(2)-std: check-stage$(1)-$(2)-std-dummy

check-stage$(1)-$(2)-rustc: check-stage$(1)-$(2)-rustc-dummy

check-stage$(1)-$(2)-cfail: check-stage$(1)-$(2)-cfail-dummy

check-stage$(1)-$(2)-rfail: check-stage$(1)-$(2)-rfail-dummy

check-stage$(1)-$(2)-rpass: check-stage$(1)-$(2)-rpass-dummy

check-stage$(1)-$(2)-bench: check-stage$(1)-$(2)-bench-dummy

check-stage$(1)-$(2)-perf: check-stage$(1)-$(2)-perf-dummy

check-stage$(1)-$(2)-pretty: check-stage$(1)-$(2)-pretty-rpass \
                        check-stage$(1)-$(2)-pretty-rfail \
                        check-stage$(1)-$(2)-pretty-bench \
                        check-stage$(1)-$(2)-pretty-pretty

check-stage$(1)-$(2)-pretty-rpass: check-stage$(1)-$(2)-pretty-rpass-dummy

check-stage$(1)-$(2)-pretty-rfail: check-stage$(1)-$(2)-pretty-rfail-dummy

check-stage$(1)-$(2)-pretty-bench: check-stage$(1)-$(2)-pretty-bench-dummy

check-stage$(1)-$(2)-pretty-pretty: check-stage$(1)-$(2)-pretty-pretty-dummy


# Rules for the standard library test runner

test/stdtest.stage$(1)-$(2)$$(X): $$(STDTEST_CRATE) $$(STDTEST_INPUTS) \
                            $$(SREQ$(1)$(2))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_$(2)) -o $$@ $$< --test

check-stage$(1)-$(2)-std-dummy: test/stdtest.stage$(1)-$(2)$$(X)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_TEST,$$<) $$(TESTARGS)


# Rules for the rustc test runner

test/rustctest.stage$(1)-$(2)$$(X): \
	$$(COMPILER_CRATE) $$(COMPILER_INPUTS) $$(SREQ$(1)$(2)) \
	$$(TARGET_LIB$(1)$(2))/$$(CFG_RUSTLLVM)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_$(2)) -o $$@ $$< --test

check-stage$(1)-$(2)-rustc-dummy: test/rustctest.stage$(1)-$(2)$$(X)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_TEST,$$<) \
	  $$(TESTARGS)


# Rules for the cfail/rfail/rpass/bench/perf test runner

CTEST_COMMON_ARGS$(1)-$(2) := --compile-lib-path $$(HOST_LIB$(1)) \
                         --run-lib-path $$(TARGET_LIB$(1)$$(CFG_HOST_TRIPLE)) \
                         --rustc-path $$(HOST_BIN$(1))/rustc$$(X) \
                         --stage-id stage$(1)-$(2) \
                         --rustcflags "$$(CFG_RUSTC_FLAGS) --target=$(2)" \
                         $$(CTEST_TESTARGS)

CFAIL_ARGS$(1)-$(2) := $$(CTEST_COMMON_ARGS$(1)-$(2)) \
                  --src-base $$(S)src/test/compile-fail/ \
                  --build-base test/compile-fail/ \
                  --mode compile-fail

RFAIL_ARGS$(1)-$(2) := $$(CTEST_COMMON_ARGS$(1)-$(2)) \
                  --src-base $$(S)src/test/run-fail/ \
                  --build-base test/run-fail/ \
                  --mode run-fail \
                  $$(CTEST_RUNTOOL)

RPASS_ARGS$(1)-$(2) := $$(CTEST_COMMON_ARGS$(1)-$(2)) \
                  --src-base $$(S)src/test/run-pass/ \
                  --build-base test/run-pass/ \
                  --mode run-pass \
                  $$(CTEST_RUNTOOL)

BENCH_ARGS$(1)-$(2) := $$(CTEST_COMMON_ARGS$(1)-$(2)) \
                  --src-base $$(S)src/test/bench/ \
                  --build-base test/bench/ \
                  --mode run-pass \
                  $$(CTEST_RUNTOOL)

PERF_ARGS$(1)-$(2) := $$(CTEST_COMMON_ARGS$(1)-$(2)) \
                  --src-base $$(S)src/test/bench/ \
                  --build-base test/perf/ \
                  --mode run-pass \
                  $$(CTEST_PERF_RUNTOOL)

PRETTY_RPASS_ARGS$(1)-$(2) := $$(CTEST_COMMON_ARGS$(1)-$(2)) \
                         --src-base $$(S)src/test/run-pass/ \
                         --build-base test/run-pass/ \
                         --mode pretty

PRETTY_RFAIL_ARGS$(1)-$(2) := $$(CTEST_COMMON_ARGS$(1)-$(2)) \
                         --src-base $$(S)src/test/run-fail/ \
                         --build-base test/run-fail/ \
                         --mode pretty

PRETTY_BENCH_ARGS$(1)-$(2) := $$(CTEST_COMMON_ARGS$(1)-$(2)) \
                         --src-base $$(S)src/test/bench/ \
                         --build-base test/bench/ \
                         --mode pretty

PRETTY_PRETTY_ARGS$(1)-$(2) := $$(CTEST_COMMON_ARGS$(1)-$(2)) \
                          --src-base $$(S)src/test/pretty/ \
                          --build-base test/pretty/ \
                          --mode pretty

check-stage$(1)-$(2)-cfail-dummy: $$(HOST_BIN$(1))/compiletest$$(X) \
	                              $$(SREQ$(1)$(2)) \
                                  $$(CFAIL_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<) $$(CFAIL_ARGS$(1)-$(2))

check-stage$(1)-$(2)-rfail-dummy: $$(HOST_BIN$(1))/compiletest$$(X) \
	                              $$(SREQ$(1)$(2)) \
                                  $$(RFAIL_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<) $$(RFAIL_ARGS$(1)-$(2))

check-stage$(1)-$(2)-rpass-dummy: $$(HOST_BIN$(1))/compiletest$$(X) \
	                              $$(SREQ$(1)$(2)) \
                                  $$(RPASS_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<) $$(RPASS_ARGS$(1)-$(2))

check-stage$(1)-$(2)-bench-dummy: $$(HOST_BIN$(1))/compiletest$$(X) \
	                              $$(SREQ$(1)$(2)) \
                                  $$(BENCH_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<) $$(BENCH_ARGS$(1)-$(2))

check-stage$(1)-$(2)-perf-dummy: $$(HOST_BIN$(1))/compiletest$$(X) \
                                 $$(SREQ$(1)$(2)) \
                                 $$(BENCH_TESTS)
	@$$(call E, perf: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<) $$(PERF_ARGS$(1)-$(2))

check-stage$(1)-$(2)-pretty-rpass-dummy: $$(HOST_BIN$(1))/compiletest$$(X) \
                                         $$(SREQ$(1)$(2)) \
                                         $$(RPASS_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<) $$(PRETTY_RPASS_ARGS$(1)-$(2))

check-stage$(1)-$(2)-pretty-rfail-dummy: $$(HOST_BIN$(1))/compiletest$$(X) \
                                         $$(SREQ$(1)$(2)) \
                                         $$(RFAIL_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<) $$(PRETTY_RFAIL_ARGS$(1)-$(2))

check-stage$(1)-$(2)-pretty-bench-dummy: $$(HOST_BIN$(1))/compiletest$$(X) \
                                         $$(SREQ$(1)$(2)) \
                                         $$(BENCH_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<) $$(PRETTY_BENCH_ARGS$(1)-$(2))

check-stage$(1)-$(2)-pretty-pretty-dummy: $$(HOST_BIN$(1))/compiletest$$(X) \
                                          $$(SREQ$(1)$(2)) \
                                          $$(PRETTY_TESTS)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<) $$(PRETTY_PRETTY_ARGS$(1)-$(2))

endef

# Instantiate the template for stage 0, 1, 2, 3

$(foreach target,$(CFG_TARGET_TRIPLES), \
 $(eval $(call TEST_STAGEN,0,$(target))) \
 $(eval $(call TEST_STAGEN,1,$(target))) \
 $(eval $(call TEST_STAGEN,2,$(target))) \
 $(eval $(call TEST_STAGEN,3,$(target))))

# Make convenient Shorthand Targets for use on command
# line or by other rules:

define DEF_CHECK_STAGE

check-stage$(1):        check-stage$(1)-$$(CFG_HOST_TRIPLE)
check-stage$(1)-perf:   check-stage$(1)-$$(CFG_HOST_TRIPLE)-perf
check-stage$(1)-rustc:  check-stage$(1)-$$(CFG_HOST_TRIPLE)-rustc
check-stage$(1)-std:    check-stage$(1)-$$(CFG_HOST_TRIPLE)-std
check-stage$(1)-rpass:  check-stage$(1)-$$(CFG_HOST_TRIPLE)-rpass
check-stage$(1)-rfail:  check-stage$(1)-$$(CFG_HOST_TRIPLE)-rfail
check-stage$(1)-cfail:  check-stage$(1)-$$(CFG_HOST_TRIPLE)-cfail
check-stage$(1)-bench:  check-stage$(1)-$$(CFG_HOST_TRIPLE)-bench

endef

$(eval $(call DEF_CHECK_STAGE,0))
$(eval $(call DEF_CHECK_STAGE,1))
$(eval $(call DEF_CHECK_STAGE,2))
$(eval $(call DEF_CHECK_STAGE,3))


######################################################################
# Fast-test rules
######################################################################

test/$(FT).rc test/$(FT_DRIVER).rs: $(TEST_RPASS_SOURCES_STAGE2) \
    $(S)src/etc/combine-tests.py
	@$(call E, check: building combined stage2 test runner)
	$(Q)$(S)src/etc/combine-tests.py

$(TARGET_HOST_LIB2)/$(FT_LIB): test/$(FT).rc $(SREQ2$(CFG_HOST_TRIPLE))
	@$(call E, compile_and_link: $@)
	$(STAGE2_$(CFG_HOST_TRIPLE)) --lib -o $@ $<

test/$(FT_DRIVER)$(X): test/$(FT_DRIVER).rs $(TARGET_HOST_LIB2)/$(FT_LIB) \
	$(SREQ2$(CFG_HOST_TRIPLE))
	@$(call E, compile_and_link: $@)
	$(STAGE2_$(CFG_HOST_TRIPLE)) -L $(HOST_LIB2) -o $@ $<

test/$(FT_DRIVER).out: test/$(FT_DRIVER)$(X) $(SREQ2$(CFG_HOST_TRIPLE))
	$(Q)$(call CFG_RUN_TEST, $<)
