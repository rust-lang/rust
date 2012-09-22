######################################################################
# Testing variables
######################################################################

RPASS_RC := $(wildcard $(S)src/test/run-pass/*.rc)
RPASS_RS := $(wildcard $(S)src/test/run-pass/*.rs)
RPASS_FULL_RC := $(wildcard $(S)src/test/run-pass-fulldeps/*.rc)
RPASS_FULL_RS := $(wildcard $(S)src/test/run-pass-fulldeps/*.rs)
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
RPASS_FULL_TESTS := $(RPASS_FULL_RC) $(RPASS_FULL_RS)
RFAIL_TESTS := $(RFAIL_RC) $(RFAIL_RS)
CFAIL_TESTS := $(CFAIL_RC) $(CFAIL_RS)
BENCH_TESTS := $(BENCH_RS)
PERF_TESTS := $(PERF_RS)
PRETTY_TESTS := $(PRETTY_RS)

FT := run_pass_stage2
FT_LIB := $(call CFG_LIB_NAME,$(FT))
FT_DRIVER := $(FT)_driver

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

# Run the compiletest runner itself under valgrind
ifdef CTEST_VALGRIND
  CFG_RUN_CTEST=$(call CFG_RUN_TEST,$(2),$(3))
else
  CFG_RUN_CTEST=$(call CFG_RUN,$(TLIB$(1)_T_$(3)_H_$(3)),$(2))
endif

# If we're running perf then set this environment variable
# to put the benchmarks into 'hard mode'
ifeq ($(MAKECMDGOALS),perf)
  RUST_BENCH=1
  export RUST_BENCH
endif


######################################################################
# Main test targets
######################################################################

.PHONY: cleantmptestlogs cleantestlibs

cleantmptestlogs:
	$(Q)rm -f tmp/*.log

cleantestlibs:
	$(Q)find $(CFG_HOST_TRIPLE)/test \
         -name '*.[odasS]' -o \
         -name '*.so' -o      \
         -name '*.dylib' -o   \
         -name '*.dll' -o     \
         -name '*.def' -o     \
         -name '*.bc' -o      \
         -name '*.dSYM' -o    \
         -name '*.libaux' -o      \
         -name '*.out' -o     \
         -name '*.err'        \
         | xargs rm -rf

check: cleantestlibs cleantmptestlogs tidy all check-stage2
	$(Q)$(S)src/etc/check-summary.py tmp/*.log

check-notidy: cleantestlibs cleantmptestlogs all check-stage2
	$(Q)$(S)src/etc/check-summary.py tmp/*.log

check-full: cleantestlibs cleantmptestlogs tidy \
            all check-stage1 check-stage2 check-stage3
	$(Q)$(S)src/etc/check-summary.py tmp/*.log

check-test: cleantestlibs cleantmptestlogs all check-stage2-rfail
	$(Q)$(S)src/etc/check-summary.py tmp/*.log

check-lite: cleantestlibs cleantmptestlogs rustc-stage2 \
	check-stage2-core check-stage2-std check-stage2-rpass \
	check-stage2-rfail check-stage2-cfail
	$(Q)$(S)src/etc/check-summary.py tmp/*.log

# Run the tidy script in multiple parts to avoid huge 'echo' commands
ifdef CFG_NOTIDY
tidy:
else

ALL_CS := $(wildcard $(S)src/rt/*.cpp \
                     $(S)src/rt/*/*.cpp \
                     $(S)src/rt/*/*/*.cpp \
                     $(S)srcrustllvm/*.cpp)
ALL_CS := $(filter-out $(S)src/rt/bigint/bigint_ext.cpp \
                       $(S)src/rt/bigint/bigint_int.cpp \
                       $(S)src/rt/miniz.cpp \
	,$(ALL_CS))
ALL_HS := $(wildcard $(S)src/rt/*.h \
                     $(S)src/rt/*/*.h \
                     $(S)src/rt/*/*/*.h \
                     $(S)srcrustllvm/*.h)
ALL_HS := $(filter-out $(S)src/rt/vg/valgrind.h \
                       $(S)src/rt/vg/memcheck.h \
                       $(S)src/rt/uthash/uthash.h \
                       $(S)src/rt/uthash/utlist.h \
                       $(S)src/rt/msvc/typeof.h \
                       $(S)src/rt/msvc/stdint.h \
                       $(S)src/rt/msvc/inttypes.h \
                       $(S)src/rt/bigint/bigint.h \
	,$(ALL_HS))

tidy:
		@$(call E, check: formatting)
		$(Q)find $(S)src -name '*.r[sc]' \
		| grep '^$(S)src/test' -v \
		| xargs -n 10 python $(S)src/etc/tidy.py
		$(Q)find $(S)src/etc -name '*.py' \
		| xargs -n 10 python $(S)src/etc/tidy.py
		$(Q)echo $(ALL_CS) \
	  	| xargs -n 10 python $(S)src/etc/tidy.py
		$(Q)echo $(ALL_HS) \
	  	| xargs -n 10 python $(S)src/etc/tidy.py

endif

######################################################################
# Extracting tests for docs
######################################################################

EXTRACT_TESTS := "$(CFG_PYTHON)" $(S)src/etc/extract-tests.py

define DEF_DOC_TEST_HOST

doc-tutorial-extract$(1):
	@$$(call E, extract: tutorial tests)
	$$(Q)rm -f $(1)/test/doc-tutorial/*.rs
	$$(Q)$$(EXTRACT_TESTS) $$(S)doc/tutorial.md $(1)/test/doc-tutorial

doc-tutorial-ffi-extract$(1):
	@$$(call E, extract: tutorial-ffi tests)
	$$(Q)rm -f $(1)/test/doc-tutorial-ffi/*.rs
	$$(Q)$$(EXTRACT_TESTS) $$(S)doc/tutorial-ffi.md $(1)/test/doc-tutorial-ffi

doc-tutorial-macros-extract$(1):
	@$$(call E, extract: tutorial-macros tests)
	$$(Q)rm -f $(1)/test/doc-tutorial-macros/*.rs
	$$(Q)$$(EXTRACT_TESTS) $$(S)doc/tutorial-macros.md $(1)/test/doc-tutorial-macros

doc-tutorial-borrowed-ptr-extract$(1):
	@$$(call E, extract: tutorial-borrowed-ptr tests)
	$$(Q)rm -f $(1)/test/doc-tutorial-borrowed-ptr/*.rs
	$$(Q)$$(EXTRACT_TESTS) $$(S)doc/tutorial-borrowed-ptr.md $(1)/test/doc-tutorial-borrowed-ptr

doc-tutorial-tasks-extract$(1):
	@$$(call E, extract: tutorial-tasks tests)
	$$(Q)rm -f $(1)/test/doc-tutorial-tasks/*.rs
	$$(Q)$$(EXTRACT_TESTS) $$(S)doc/tutorial-tasks.md $(1)/test/doc-tutorial-tasks

doc-ref-extract$(1):
	@$$(call E, extract: ref tests)
	$$(Q)rm -f $(1)/test/doc-ref/*.rs
	$$(Q)$$(EXTRACT_TESTS) $$(S)doc/rust.md $(1)/test/doc-ref

endef

$(foreach host,$(CFG_TARGET_TRIPLES), \
 $(eval $(call DEF_DOC_TEST_HOST,$(host))))

######################################################################
# Rules for the test runners
######################################################################

define TEST_STAGEN

# All the per-stage build rules you might want to call from the
# command line.
#
# $(1) is the stage number
# $(2) is the target triple to test
# $(3) is the host triple to test

# Prerequisites for compiletest tests
TEST_SREQ$(1)_T_$(2)_H_$(3) = \
	$$(HBIN$(1)_H_$(3))/compiletest$$(X) \
	$$(SREQ$(1)_T_$(2)_H_$(3))

# Prerequisites for compiletest tests that have deps on librustc, etc
FULL_TEST_SREQ$(1)_T_$(2)_H_$(3) = \
	$$(HBIN$(1)_H_$(3))/compiletest$$(X) \
	$$(SREQ$(1)_T_$(2)_H_$(3)) \
	$$(TLIBRUSTC_DEFAULT$(1)_T_$(2)_H_$(3))

check-stage$(1)-T-$(2)-H-$(3):     				\
	check-stage$(1)-T-$(2)-H-$(3)-rustc			\
	check-stage$(1)-T-$(2)-H-$(3)-core          \
	check-stage$(1)-T-$(2)-H-$(3)-std			\
	check-stage$(1)-T-$(2)-H-$(3)-rpass			\
	check-stage$(1)-T-$(2)-H-$(3)-rpass-full			\
	check-stage$(1)-T-$(2)-H-$(3)-rfail			\
	check-stage$(1)-T-$(2)-H-$(3)-cfail			\
	check-stage$(1)-T-$(2)-H-$(3)-bench			\
	check-stage$(1)-T-$(2)-H-$(3)-pretty        \
    check-stage$(1)-T-$(2)-H-$(3)-rustdoc       \
    check-stage$(1)-T-$(2)-H-$(3)-cargo       \
    check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial  \
    check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-ffi  \
    check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-macros  \
    check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-borrowed-ptr  \
    check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-tasks  \
    check-stage$(1)-T-$(2)-H-$(3)-doc-ref

check-stage$(1)-T-$(2)-H-$(3)-core:				\
	check-stage$(1)-T-$(2)-H-$(3)-core-dummy

check-stage$(1)-T-$(2)-H-$(3)-std:				\
	check-stage$(1)-T-$(2)-H-$(3)-std-dummy

check-stage$(1)-T-$(2)-H-$(3)-rustc:				\
	check-stage$(1)-T-$(2)-H-$(3)-rustc-dummy

check-stage$(1)-T-$(2)-H-$(3)-cfail:				\
	check-stage$(1)-T-$(2)-H-$(3)-cfail-dummy

check-stage$(1)-T-$(2)-H-$(3)-rfail:				\
	check-stage$(1)-T-$(2)-H-$(3)-rfail-dummy

check-stage$(1)-T-$(2)-H-$(3)-rpass:				\
	check-stage$(1)-T-$(2)-H-$(3)-rpass-dummy

check-stage$(1)-T-$(2)-H-$(3)-rpass-full:				\
	check-stage$(1)-T-$(2)-H-$(3)-rpass-full-dummy

check-stage$(1)-T-$(2)-H-$(3)-bench:				\
	check-stage$(1)-T-$(2)-H-$(3)-bench-dummy

check-stage$(1)-T-$(2)-H-$(3)-perf:				\
	check-stage$(1)-T-$(2)-H-$(3)-perf-dummy

check-stage$(1)-T-$(2)-H-$(3)-pretty:			\
	check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass	\
	check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-full	\
    check-stage$(1)-T-$(2)-H-$(3)-pretty-rfail	\
    check-stage$(1)-T-$(2)-H-$(3)-pretty-bench	\
    check-stage$(1)-T-$(2)-H-$(3)-pretty-pretty

check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass:			\
	check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-dummy

check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-full:			\
	check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-full-dummy

check-stage$(1)-T-$(2)-H-$(3)-pretty-rfail:			\
	check-stage$(1)-T-$(2)-H-$(3)-pretty-rfail-dummy

check-stage$(1)-T-$(2)-H-$(3)-pretty-bench:			\
	check-stage$(1)-T-$(2)-H-$(3)-pretty-bench-dummy

check-stage$(1)-T-$(2)-H-$(3)-pretty-pretty:				\
	check-stage$(1)-T-$(2)-H-$(3)-pretty-pretty-dummy

check-stage$(1)-T-$(2)-H-$(3)-rustdoc:				\
	check-stage$(1)-T-$(2)-H-$(3)-rustdoc-dummy

check-stage$(1)-T-$(2)-H-$(3)-cargo:				\
	check-stage$(1)-T-$(2)-H-$(3)-cargo-dummy

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial: \
	check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-dummy

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-ffi: \
	check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-ffi-dummy

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-macros: \
	check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-macros-dummy

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-borrowed-ptr: \
	check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-borrowed-ptr-dummy

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-tasks: \
	check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-tasks-dummy

check-stage$(1)-T-$(2)-H-$(3)-doc-ref: \
	check-stage$(1)-T-$(2)-H-$(3)-doc-ref-dummy

# Rules for the core library test runner

$(3)/test/coretest.stage$(1)-$(2)$$(X):			\
		$$(CORELIB_CRATE) $$(CORELIB_INPUTS)	\
        $$(SREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< --test

check-stage$(1)-T-$(2)-H-$(3)-core-dummy:			\
		$(3)/test/coretest.stage$(1)-$(2)$$(X)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_TEST,$$<,$(2),$(3)) $$(TESTARGS)	\
	--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-core.log

# Rules for the standard library test runner

$(3)/test/stdtest.stage$(1)-$(2)$$(X):			\
		$$(STDLIB_CRATE) $$(STDLIB_INPUTS)	\
        $$(SREQ$(1)_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< --test

check-stage$(1)-T-$(2)-H-$(3)-std-dummy:			\
		$(3)/test/stdtest.stage$(1)-$(2)$$(X)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_TEST,$$<,$(2),$(3)) $$(TESTARGS)	\
	--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-std.log

# Rules for the rustc test runner

$(3)/test/rustctest.stage$(1)-$(2)$$(X):					\
		$$(COMPILER_CRATE)									\
		$$(COMPILER_INPUTS)									\
		$$(SREQ$(1)_T_$(2)_H_$(3))							\
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_RUSTLLVM)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< --test

check-stage$(1)-T-$(2)-H-$(3)-rustc-dummy:		\
		$(3)/test/rustctest.stage$(1)-$(2)$$(X)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_TEST,$$<,$(2),$(3)) $$(TESTARGS)   \
	--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-rustc.log

# Rules for the rustdoc test runner

$(3)/test/rustdoctest.stage$(1)-$(2)$$(X):					\
		$$(RUSTDOC_CRATE) $$(RUSTDOC_INPUTS)		\
		$$(TSREQ$(1)_T_$(2)_H_$(3))					\
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_CORELIB)  \
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_STDLIB)   \
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< --test

check-stage$(1)-T-$(2)-H-$(3)-rustdoc-dummy:		\
		$(3)/test/rustdoctest.stage$(1)-$(2)$$(X)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_TEST,$$<,$(2),$(3)) $$(TESTARGS)	\
	--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-rustdoc.log

# Rules for the cargo test runner

$(3)/test/cargotest.stage$(1)-$(2)$$(X):					\
		$$(CARGO_CRATE) $$(CARGO_INPUTS)		\
		$$(TSREQ$(1)_T_$(2)_H_$(3))					\
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_CORELIB)  \
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_STDLIB)   \
		$$(TLIB$(1)_T_$(2)_H_$(3))/$$(CFG_LIBRUSTC)
	@$$(call E, compile_and_link: $$@)
	$$(STAGE$(1)_T_$(2)_H_$(3)) -o $$@ $$< --test

check-stage$(1)-T-$(2)-H-$(3)-cargo-dummy:		\
		$(3)/test/cargotest.stage$(1)-$(2)$$(X)
	@$$(call E, run: $$<)
	$$(Q)$$(call CFG_RUN_TEST,$$<,$(2),$(3)) $$(TESTARGS)	\
	--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-cargo.log

# Rules for the cfail/rfail/rpass/bench/perf test runner

CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3) :=						\
		--compile-lib-path $$(HLIB$(1)_H_$(3))				\
        --run-lib-path $$(TLIB$(1)_T_$(2)_H_$(3))			\
        --rustc-path $$(HBIN$(1)_H_$(3))/rustc$$(X)			\
        --aux-base $$(S)src/test/auxiliary/                 \
        --stage-id stage$(1)-$(2)							\
        --rustcflags "$$(CFG_RUSTC_FLAGS) --target=$(2)"	\
        $$(CTEST_TESTARGS)

CFAIL_ARGS$(1)-T-$(2)-H-$(3) :=					\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/compile-fail/	\
        --build-base $(3)/test/compile-fail/	\
        --mode compile-fail

RFAIL_ARGS$(1)-T-$(2)-H-$(3) :=					\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/run-fail/		\
        --build-base $(3)/test/run-fail/		\
        --mode run-fail							\
        $$(CTEST_RUNTOOL)

RPASS_ARGS$(1)-T-$(2)-H-$(3) :=				\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/run-pass/		\
        --build-base $(3)/test/run-pass/		\
        --mode run-pass					\
        $$(CTEST_RUNTOOL)

RPASS_FULL_ARGS$(1)-T-$(2)-H-$(3) :=				\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/run-pass-fulldeps/		\
        --build-base $(3)/test/run-pass-fulldeps/		\
        --mode run-pass					\
        $$(CTEST_RUNTOOL)

BENCH_ARGS$(1)-T-$(2)-H-$(3) :=				\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/bench/			\
        --build-base $(3)/test/bench/			\
        --mode run-pass					\
        $$(CTEST_RUNTOOL)

PERF_ARGS$(1)-T-$(2)-H-$(3) :=					\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/bench/			\
        --build-base $(3)/test/perf/			\
        --mode run-pass							\
        $$(CTEST_PERF_RUNTOOL)

PRETTY_RPASS_ARGS$(1)-T-$(2)-H-$(3) :=			\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/run-pass/		\
        --build-base $(3)/test/run-pass/		\
        --mode pretty

PRETTY_RPASS_FULL_ARGS$(1)-T-$(2)-H-$(3) :=			\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/run-pass-fulldeps/		\
        --build-base $(3)/test/run-pass-fulldeps/		\
        --mode pretty

PRETTY_RFAIL_ARGS$(1)-T-$(2)-H-$(3) :=			\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/run-fail/		\
        --build-base $(3)/test/run-fail/		\
        --mode pretty

PRETTY_BENCH_ARGS$(1)-T-$(2)-H-$(3) :=			\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/bench/			\
        --build-base $(3)/test/bench/			\
        --mode pretty

PRETTY_PRETTY_ARGS$(1)-T-$(2)-H-$(3) :=			\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $$(S)src/test/pretty/		\
        --build-base $(3)/test/pretty/			\
        --mode pretty

DOC_TUTORIAL_ARGS$(1)-T-$(2)-H-$(3) :=			\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $(3)/test/doc-tutorial/		\
        --build-base $(3)/test/doc-tutorial/		\
        --mode run-pass

DOC_TUTORIAL_FFI_ARGS$(1)-T-$(2)-H-$(3) :=		\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $(3)/test/doc-tutorial-ffi/		\
        --build-base $(3)/test/doc-tutorial-ffi/	\
        --mode run-pass

DOC_TUTORIAL_MACROS_ARGS$(1)-T-$(2)-H-$(3) :=		\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $(3)/test/doc-tutorial-macros/	\
        --build-base $(3)/test/doc-tutorial-macros/	\
        --mode run-pass

DOC_TUTORIAL_BORROWED_PTR_ARGS$(1)-T-$(2)-H-$(3) :=	\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $(3)/test/doc-tutorial-borrowed-ptr/	\
        --build-base $(3)/test/doc-tutorial-borrowed-ptr/ \
        --mode run-pass

DOC_TUTORIAL_TASKS_ARGS$(1)-T-$(2)-H-$(3) :=	\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $(3)/test/doc-tutorial-tasks/	\
        --build-base $(3)/test/doc-tutorial-tasks/ \
        --mode run-pass

DOC_REF_ARGS$(1)-T-$(2)-H-$(3) :=			\
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3))	\
        --src-base $(3)/test/doc-ref/			\
        --build-base $(3)/test/doc-ref/			\
        --mode run-pass

check-stage$(1)-T-$(2)-H-$(3)-cfail-dummy:		\
		$$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
	        $$(CFAIL_TESTS)
	@$$(call E, run cfail: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(CFAIL_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-cfail.log

check-stage$(1)-T-$(2)-H-$(3)-rfail-dummy:		\
		$$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
		$$(RFAIL_TESTS)
	@$$(call E, run rfail: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(RFAIL_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-rfail.log

check-stage$(1)-T-$(2)-H-$(3)-rpass-dummy:		\
		$$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
	        $$(RPASS_TESTS)
	@$$(call E, run rpass-full: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(RPASS_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-rpass.log

check-stage$(1)-T-$(2)-H-$(3)-rpass-full-dummy:		\
		$$(FULL_TEST_SREQ$(1)_T_$(2)_H_$(3))		\
	        $$(RPASS_FULL_TESTS)
	@$$(call E, run rpass: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(RPASS_FULL_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-rpass-full.log

check-stage$(1)-T-$(2)-H-$(3)-bench-dummy:		\
		$$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
		$$(BENCH_TESTS)
	@$$(call E, run bench: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(BENCH_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-bench.log

check-stage$(1)-T-$(2)-H-$(3)-perf-dummy:		\
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
	        $$(BENCH_TESTS)
	@$$(call E, perf: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(PERF_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-perf.log

check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-dummy:	\
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
	        $$(RPASS_TESTS)
	@$$(call E, run pretty-rpass: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(PRETTY_RPASS_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass.log

check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-full-dummy:	\
	        $$(FULL_TEST_SREQ$(1)_T_$(2)_H_$(3))		\
	        $$(RPASS_FULL_TESTS)
	@$$(call E, run pretty-rpass-full: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(PRETTY_RPASS_FULL_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-full.log

check-stage$(1)-T-$(2)-H-$(3)-pretty-rfail-dummy:	\
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
	        $$(RFAIL_TESTS)
	@$$(call E, run pretty-rfail: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(PRETTY_RFAIL_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-pretty-rfail.log

check-stage$(1)-T-$(2)-H-$(3)-pretty-bench-dummy:	\
		$$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
	        $$(BENCH_TESTS)
	@$$(call E, run pretty-bench: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(PRETTY_BENCH_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-pretty-bench.log

check-stage$(1)-T-$(2)-H-$(3)-pretty-pretty-dummy:	\
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
	        $$(PRETTY_TESTS)
	@$$(call E, run pretty-pretty: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
		$$(PRETTY_PRETTY_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-pretty-pretty.log

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-dummy:       \
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
                doc-tutorial-extract$(3)
	@$$(call E, run doc-tutorial: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
                $$(DOC_TUTORIAL_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial.log

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-ffi-dummy:       \
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
                doc-tutorial-ffi-extract$(3)
	@$$(call E, run doc-tutorial-ffi: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
                $$(DOC_TUTORIAL_FFI_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-ffi.log

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-macros-dummy:       \
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
                doc-tutorial-macros-extract$(3)
	@$$(call E, run doc-tutorial-macros: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
                $$(DOC_TUTORIAL_MACROS_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-macros.log

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-borrowed-ptr-dummy:       \
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
                doc-tutorial-borrowed-ptr-extract$(3)
	@$$(call E, run doc-tutorial-borrowed-ptr: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
                $$(DOC_TUTORIAL_BORROWED_PTR_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-borrowed-ptr.log

check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-tasks-dummy:       \
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
                doc-tutorial-tasks-extract$(3)
	@$$(call E, run doc-tutorial-tasks: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
                $$(DOC_TUTORIAL_TASKS_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-doc-tutorial-tasks.log

check-stage$(1)-T-$(2)-H-$(3)-doc-ref-dummy:            \
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3))		\
                doc-ref-extract$(3)
	@$$(call E, run doc-ref: $$<)
	$$(Q)$$(call CFG_RUN_CTEST,$(1),$$<,$(3)) \
                $$(DOC_REF_ARGS$(1)-T-$(2)-H-$(3)) \
		--logfile tmp/check-stage$(1)-T-$(2)-H-$(3)-doc-ref.log

endef

# Instantiate the template for stage 0, 1, 2, 3

$(foreach host,$(CFG_TARGET_TRIPLES), \
 $(eval $(foreach target,$(CFG_TARGET_TRIPLES), \
  $(eval $(foreach stage,$(STAGES), \
   $(eval $(call TEST_STAGEN,$(stage),$(target),$(host))))))))

######################################################################
# Fast-test rules
######################################################################

GENERATED += tmp/$(FT).rc tmp/$(FT_DRIVER).rs

tmp/$(FT).rc tmp/$(FT_DRIVER).rs: \
		$(RPASS_TESTS) \
		$(S)src/etc/combine-tests.py
	@$(call E, check: building combined stage2 test runner)
	$(Q)$(S)src/etc/combine-tests.py

define DEF_CHECK_FAST_FOR_T_H
# $(1) unused
# $(2) target triple
# $(3) host triple

$$(TLIB2_T_$(2)_H_$(3))/$$(FT_LIB): \
		tmp/$$(FT).rc \
		$$(SREQ2_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@)
	$$(STAGE2_T_$(2)_H_$(3)) --lib -o $$@ $$<

$(3)/test/$$(FT_DRIVER)-$(2)$$(X): \
		tmp/$$(FT_DRIVER).rs \
		$$(TLIB2_T_$(2)_H_$(3))/$$(FT_LIB) \
		$$(SREQ2_T_$(2)_H_$(3))
	@$$(call E, compile_and_link: $$@ $$<)
	$$(STAGE2_T_$(2)_H_$(3)) -o $$@ $$<

$(3)/test/$$(FT_DRIVER)-$(2).out: \
		$(3)/test/$$(FT_DRIVER)-$(2)$$(X) \
		$$(SREQ2_T_$(2)_H_$(3))
	$$(Q)$$(call CFG_RUN_TEST,$$<,$(2),$(3)) \
	--logfile tmp/$$(FT_DRIVER)-$(2).log

check-fast-T-$(2)-H-$(3):     			\
	check-stage2-T-$(2)-H-$(3)-rustc	\
	check-stage2-T-$(2)-H-$(3)-core		\
	check-stage2-T-$(2)-H-$(3)-std		\
	$(3)/test/$$(FT_DRIVER)-$(2).out

endef

$(foreach host,$(CFG_TARGET_TRIPLES), \
 $(eval $(foreach target,$(CFG_TARGET_TRIPLES), \
   $(eval $(call DEF_CHECK_FAST_FOR_T_H,,$(target),$(host))))))

######################################################################
# Shortcut rules
######################################################################

define DEF_CHECK_FOR_STAGE_H

check-stage$(1)-H-$(2):					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2))
check-stage$(1)-H-$(2)-perf:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-perf)
check-stage$(1)-H-$(2)-rustc:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-rustc)
check-stage$(1)-H-$(2)-core:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-core)
check-stage$(1)-H-$(2)-std:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-std)
check-stage$(1)-H-$(2)-rpass:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-rpass)
check-stage$(1)-H-$(2)-rpass-full:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-rpass-full)
check-stage$(1)-H-$(2)-rfail:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-rfail)
check-stage$(1)-H-$(2)-cfail:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-cfail)
check-stage$(1)-H-$(2)-bench:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-bench)
check-stage$(1)-H-$(2)-pretty:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-pretty)
check-stage$(1)-H-$(2)-pretty-rpass:				\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-pretty-rpass)
check-stage$(1)-H-$(2)-pretty-rpass-full:				\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-pretty-rpass-full)
check-stage$(1)-H-$(2)-pretty-rfail:				\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-pretty-rfail)
check-stage$(1)-H-$(2)-pretty-bench:				\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-pretty-bench)
check-stage$(1)-H-$(2)-pretty-pretty:				\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-pretty-pretty)
check-stage$(1)-H-$(2)-rustdoc:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-rustdoc)
check-stage$(1)-H-$(2)-cargo:					\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-cargo)
check-stage$(1)-H-$(2)-doc-tutorial:				\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-doc-tutorial)
check-stage$(1)-H-$(2)-doc-tutorial-ffi:			\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-doc-tutorial-ffi)
check-stage$(1)-H-$(2)-doc-tutorial-macros:			\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-doc-tutorial-macros)
check-stage$(1)-H-$(2)-doc-tutorial-borrowed-ptr:		\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-doc-tutorial-borrowed-ptr)
check-stage$(1)-H-$(2)-doc-tutorial-tasks:		\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-doc-tutorial-tasks)
check-stage$(1)-H-$(2)-doc-ref:				\
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-T-$$(target)-H-$(2)-doc-ref)

endef

$(foreach stage,$(STAGES),					\
 $(eval $(foreach target,$(CFG_TARGET_TRIPLES),			\
  $(eval $(call DEF_CHECK_FOR_STAGE_H,$(stage),$(target))))))

define DEF_CHECK_FAST_FOR_H

check-fast-H-$(1): 		check-fast-T-$(1)-H-$(1)

endef

$(foreach target,$(CFG_TARGET_TRIPLES),			\
 $(eval $(call DEF_CHECK_FAST_FOR_H,$(target))))

define DEF_CHECK_ALL_FOR_STAGE

check-stage$(1)-H-all: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target))
check-stage$(1)-H-all-perf: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-perf)
check-stage$(1)-H-all-rustc: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-rustc)
check-stage$(1)-H-all-core: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-core)
check-stage$(1)-H-all-std: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-std)
check-stage$(1)-H-all-rpass: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-rpass)
check-stage$(1)-H-all-rpass-full: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-rpass-full)
check-stage$(1)-H-all-rfail: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-rfail)
check-stage$(1)-H-all-cfail: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-cfail)
check-stage$(1)-H-all-bench: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-bench)
check-stage$(1)-H-all-pretty: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-pretty)
check-stage$(1)-H-all-pretty-rpass: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-pretty-rpass)
check-stage$(1)-H-all-pretty-rpass-full: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-pretty-rpass-full)
check-stage$(1)-H-all-pretty-rfail: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-pretty-rfail)
check-stage$(1)-H-all-pretty-bench: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-pretty-bench)
check-stage$(1)-H-all-pretty-pretty: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-pretty-pretty)
check-stage$(1)-H-all-rustdoc: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-rustdoc)
check-stage$(1)-H-all-cargo: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-cargo)
check-stage$(1)-H-all-doc-tutorial: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-doc-tutorial)
check-stage$(1)-H-all-doc-ref: \
	$$(foreach target,$$(CFG_TARGET_TRIPLES),	\
	 check-stage$(1)-H-$$(target)-doc-ref)

endef

$(foreach stage,$(STAGES),						\
 $(eval $(call DEF_CHECK_ALL_FOR_STAGE,$(stage))))

define DEF_CHECK_FOR_STAGE

check-stage$(1): check-stage$(1)-H-$$(CFG_HOST_TRIPLE)
check-stage$(1)-perf: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-perf
check-stage$(1)-rustc: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-rustc
check-stage$(1)-core: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-core
check-stage$(1)-std: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-std
check-stage$(1)-rpass: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-rpass
check-stage$(1)-rpass-full: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-rpass-full
check-stage$(1)-rfail: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-rfail
check-stage$(1)-cfail: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-cfail
check-stage$(1)-bench: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-bench
check-stage$(1)-pretty: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-pretty
check-stage$(1)-pretty-rpass: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-pretty-rpass
check-stage$(1)-pretty-rpass-full: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-pretty-rpass-full
check-stage$(1)-pretty-rfail: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-pretty-rfail
check-stage$(1)-pretty-bench: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-pretty-bench
check-stage$(1)-pretty-pretty: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-pretty-pretty
check-stage$(1)-rustdoc: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-rustdoc
check-stage$(1)-cargo: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-cargo
check-stage$(1)-doc-tutorial: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-doc-tutorial
check-stage$(1)-doc-tutorial-ffi: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-doc-tutorial-ffi
check-stage$(1)-doc-tutorial-macros: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-doc-tutorial-macros
check-stage$(1)-doc-tutorial-borrowed-ptr: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-doc-tutorial-borrowed-ptr
check-stage$(1)-doc-tutorial-tasks: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-doc-tutorial-tasks
check-stage$(1)-doc-ref: check-stage$(1)-H-$$(CFG_HOST_TRIPLE)-doc-ref

endef

$(foreach stage,$(STAGES),						\
 $(eval $(call DEF_CHECK_FOR_STAGE,$(stage))))

check-fast: tidy check-fast-H-$(CFG_HOST_TRIPLE)
