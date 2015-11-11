# Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.


######################################################################
# Test variables
######################################################################

# The names of crates that must be tested

# libcore/librustc_unicode tests are in a separate crate
DEPS_coretest :=
$(eval $(call RUST_CRATE,coretest))

DEPS_collectionstest :=
$(eval $(call RUST_CRATE,collectionstest))

TEST_TARGET_CRATES = $(filter-out core rustc_unicode alloc_system libc \
		     		  alloc_jemalloc,$(TARGET_CRATES)) \
			collectionstest coretest
TEST_DOC_CRATES = $(DOC_CRATES)
TEST_HOST_CRATES = $(filter-out rustc_typeck rustc_borrowck rustc_resolve \
		   		rustc_trans rustc_lint,\
                     $(HOST_CRATES))
TEST_CRATES = $(TEST_TARGET_CRATES) $(TEST_HOST_CRATES)

######################################################################
# Environment configuration
######################################################################

# The arguments to all test runners
ifdef TESTNAME
  TESTARGS += $(TESTNAME)
endif

ifdef CHECK_IGNORED
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

# --bench is only relevant for crate tests, not for the compile tests
ifdef PLEASE_BENCH
  TESTARGS += --bench
endif

ifdef VERBOSE
  CTEST_TESTARGS += --verbose
endif

# Setting locale ensures that gdb's output remains consistent.
# This prevents tests from failing with some locales (fixes #17423).
export LC_ALL=C

# If we're running perf then set this environment variable
# to put the benchmarks into 'hard mode'
ifeq ($(MAKECMDGOALS),perf)
  export RUST_BENCH=1
endif

TEST_LOG_FILE=tmp/check-stage$(1)-T-$(2)-H-$(3)-$(4).log
TEST_OK_FILE=tmp/check-stage$(1)-T-$(2)-H-$(3)-$(4).ok

define DEF_TARGET_COMMANDS

ifdef CFG_UNIXY_$(1)
  CFG_RUN_TEST_$(1)=$$(TARGET_RPATH_VAR$$(2)_T_$$(3)_H_$$(4)) \
	  $$(call CFG_RUN_$(1),,$$(CFG_VALGRIND) $$(1))
endif

ifdef CFG_WINDOWSY_$(1)
  CFG_TESTLIB_$(1)=$$(CFG_BUILD_DIR)$$(2)/$$(strip \
   $$(if $$(findstring stage0,$$(1)), \
       stage0/$$(CFG_LIBDIR_RELATIVE), \
      $$(if $$(findstring stage1,$$(1)), \
           stage1/$$(CFG_LIBDIR_RELATIVE), \
          $$(if $$(findstring stage2,$$(1)), \
               stage2/$$(CFG_LIBDIR_RELATIVE), \
               $$(if $$(findstring stage3,$$(1)), \
                    stage3/$$(CFG_LIBDIR_RELATIVE), \
               )))))/rustlib/$$(CFG_BUILD)/lib
  CFG_RUN_TEST_$(1)=$$(TARGET_RPATH_VAR$$(2)_T_$$(3)_H_$$(4)) \
	  $$(call CFG_RUN_$(1),$$(call CFG_TESTLIB_$(1),$$(1),$$(4)),$$(1))
endif

# Run the compiletest runner itself under valgrind
ifdef CTEST_VALGRIND
CFG_RUN_CTEST_$(1)=$$(RPATH_VAR$$(1)_T_$$(3)_H_$$(3)) \
      $$(call CFG_RUN_TEST_$$(CFG_BUILD),$$(3),$$(4))
else
CFG_RUN_CTEST_$(1)=$$(RPATH_VAR$$(1)_T_$$(3)_H_$$(3)) \
      $$(call CFG_RUN_$$(CFG_BUILD),$$(TLIB$$(1)_T_$$(3)_H_$$(3)),$$(2))
endif

endef

$(foreach target,$(CFG_TARGET), \
  $(eval $(call DEF_TARGET_COMMANDS,$(target))))

# Target platform specific variables for android
define DEF_ADB_DEVICE_STATUS
CFG_ADB_DEVICE_STATUS=$(1)
endef

$(foreach target,$(CFG_TARGET), \
  $(if $(findstring android, $(target)), \
    $(if $(findstring adb,$(CFG_ADB)), \
      $(if $(findstring device,$(shell $(CFG_ADB) devices 2>/dev/null | grep -E '^[:_A-Za-z0-9-]+[[:blank:]]+device')), \
        $(info check: android device attached) \
        $(eval $(call DEF_ADB_DEVICE_STATUS, true)), \
        $(info check: android device not attached) \
        $(eval $(call DEF_ADB_DEVICE_STATUS, false)) \
      ), \
      $(info check: adb not found) \
      $(eval $(call DEF_ADB_DEVICE_STATUS, false)) \
    ), \
  ) \
)

ifeq ($(CFG_ADB_DEVICE_STATUS),true)
CFG_ADB_TEST_DIR=/data/tmp

$(info check: android device test dir $(CFG_ADB_TEST_DIR) ready \
 $(shell $(CFG_ADB) remount 1>/dev/null) \
 $(shell $(CFG_ADB) shell rm -r $(CFG_ADB_TEST_DIR) >/dev/null) \
 $(shell $(CFG_ADB) shell mkdir $(CFG_ADB_TEST_DIR)) \
 $(shell $(CFG_ADB) push $(S)src/etc/adb_run_wrapper.sh $(CFG_ADB_TEST_DIR) 1>/dev/null) \
 $(foreach target,$(CFG_TARGET), \
  $(if $(findstring android, $(target)), \
   $(shell $(CFG_ADB) shell mkdir $(CFG_ADB_TEST_DIR)/$(target)) \
   $(foreach crate,$(TARGET_CRATES), \
    $(shell $(CFG_ADB) push $(TLIB2_T_$(target)_H_$(CFG_BUILD))/$(call CFG_LIB_GLOB_$(target),$(crate)) \
                    $(CFG_ADB_TEST_DIR)/$(target))), \
 )))
else
CFG_ADB_TEST_DIR=
endif

# $(1) - name of doc test
# $(2) - file of the test
define DOCTEST
DOC_NAMES := $$(DOC_NAMES) $(1)
DOCFILE_$(1) := $(2)
endef

$(foreach doc,$(DOCS), \
  $(eval $(call DOCTEST,md-$(doc),$(S)src/doc/$(doc).md)))
$(foreach file,$(wildcard $(S)src/doc/trpl/*.md), \
  $(eval $(call DOCTEST,$(file:$(S)src/doc/trpl/%.md=trpl-%),$(file))))
$(foreach file,$(wildcard $(S)src/doc/nomicon/*.md), \
  $(eval $(call DOCTEST,$(file:$(S)src/doc/nomicon/%.md=nomicon-%),$(file))))
######################################################################
# Main test targets
######################################################################

# The main testing target. Tests lots of stuff.
check: check-sanitycheck cleantmptestlogs cleantestlibs all check-stage2 tidy
	$(Q)$(CFG_PYTHON) $(S)src/etc/check-summary.py tmp/*.log

# As above but don't bother running tidy.
check-notidy: check-sanitycheck cleantmptestlogs cleantestlibs all check-stage2
	$(Q)$(CFG_PYTHON) $(S)src/etc/check-summary.py tmp/*.log

# A slightly smaller set of tests for smoke testing.
check-lite: check-sanitycheck cleantestlibs cleantmptestlogs \
	$(foreach crate,$(TEST_TARGET_CRATES),check-stage2-$(crate)) \
	check-stage2-rpass check-stage2-rpass-valgrind \
	check-stage2-rfail check-stage2-cfail check-stage2-pfail check-stage2-rmake
	$(Q)$(CFG_PYTHON) $(S)src/etc/check-summary.py tmp/*.log

# Only check the 'reference' tests: rpass/cfail/rfail/rmake.
check-ref: check-sanitycheck cleantestlibs cleantmptestlogs check-stage2-rpass \
	check-stage2-rpass-valgrind check-stage2-rfail check-stage2-cfail check-stage2-pfail \
	check-stage2-rmake
	$(Q)$(CFG_PYTHON) $(S)src/etc/check-summary.py tmp/*.log

# Only check the docs.
check-docs: check-sanitycheck cleantestlibs cleantmptestlogs check-stage2-docs
	$(Q)$(CFG_PYTHON) $(S)src/etc/check-summary.py tmp/*.log

# Some less critical tests that are not prone to breakage.
# Not run as part of the normal test suite, but tested by bors on checkin.
check-secondary: check-build-compiletest check-build-lexer-verifier check-lexer check-pretty

.PHONY: check-sanitycheck

check-sanitycheck:
	$(Q)$(CFG_PYTHON) $(S)src/etc/check-sanitycheck.py

# check + check-secondary.
#
# Issue #17883: build check-secondary first so hidden dependencies in
# e.g. building compiletest are exercised (resolve those by adding
# deps to rules that need them; not by putting `check` first here).
check-all: check-secondary check

# Pretty-printing tests.
check-pretty: check-stage2-T-$(CFG_BUILD)-H-$(CFG_BUILD)-pretty-exec

define DEF_CHECK_BUILD_COMPILETEST_FOR_STAGE
check-stage$(1)-build-compiletest: 	$$(HBIN$(1)_H_$(CFG_BUILD))/compiletest$$(X_$(CFG_BUILD))
endef

$(foreach stage,$(STAGES), \
 $(eval $(call DEF_CHECK_BUILD_COMPILETEST_FOR_STAGE,$(stage))))

check-build-compiletest: \
	check-stage1-build-compiletest \
	check-stage2-build-compiletest

.PHONY: cleantmptestlogs cleantestlibs

cleantmptestlogs:
	$(Q)rm -f tmp/*.log

cleantestlibs:
	$(Q)find $(CFG_BUILD)/test \
         -name '*.[odasS]' -o \
         -name '*.so' -o \
         -name '*.dylib' -o \
         -name '*.dll' -o \
         -name '*.def' -o \
         -name '*.bc' -o \
         -name '*.dSYM' -o \
         -name '*.libaux' -o \
         -name '*.out' -o \
         -name '*.err' -o \
	 -name '*.debugger.script' \
         | xargs rm -rf


######################################################################
# Tidy
######################################################################

ifdef CFG_NOTIDY
.PHONY: tidy
tidy:
else

# Run the tidy script in multiple parts to avoid huge 'echo' commands
.PHONY: tidy
tidy: tidy-basic tidy-binaries tidy-errors tidy-features

endif

.PHONY: tidy-basic
tidy-basic:
		@$(call E, check: formatting)
		$(Q) $(CFG_PYTHON) $(S)src/etc/tidy.py $(S)src/

.PHONY: tidy-binaries
tidy-binaries:
		@$(call E, check: binaries)
		$(Q)find $(S)src -type f \
		    \( -perm -u+x -or -perm -g+x -or -perm -o+x \) \
		    -not -name '*.rs' -and -not -name '*.py' \
		    -and -not -name '*.sh' -and -not -name '*.pp' \
		| grep '^$(S)src/jemalloc' -v \
		| grep '^$(S)src/libuv' -v \
		| grep '^$(S)src/llvm' -v \
		| grep '^$(S)src/rt/hoedown' -v \
		| grep '^$(S)src/gyp' -v \
		| grep '^$(S)src/etc' -v \
		| grep '^$(S)src/doc' -v \
		| grep '^$(S)src/compiler-rt' -v \
		| grep '^$(S)src/libbacktrace' -v \
		| grep '^$(S)src/rust-installer' -v \
		| grep '^$(S)src/liblibc' -v \
		| xargs $(CFG_PYTHON) $(S)src/etc/check-binaries.py

.PHONY: tidy-errors
tidy-errors:
		@$(call E, check: extended errors)
		$(Q) $(CFG_PYTHON) $(S)src/etc/errorck.py $(S)src/

.PHONY: tidy-features
tidy-features:
		@$(call E, check: feature sanity)
		$(Q) $(CFG_PYTHON) $(S)src/etc/featureck.py $(S)src/


######################################################################
# Sets of tests
######################################################################

define DEF_TEST_SETS

check-stage$(1)-T-$(2)-H-$(3)-exec: \
	check-stage$(1)-T-$(2)-H-$(3)-rpass-exec \
	check-stage$(1)-T-$(2)-H-$(3)-rfail-exec \
	check-stage$(1)-T-$(2)-H-$(3)-cfail-exec \
	check-stage$(1)-T-$(2)-H-$(3)-pfail-exec \
    check-stage$(1)-T-$(2)-H-$(3)-rpass-valgrind-exec \
    check-stage$(1)-T-$(2)-H-$(3)-rpass-full-exec \
    check-stage$(1)-T-$(2)-H-$(3)-rfail-full-exec \
	check-stage$(1)-T-$(2)-H-$(3)-cfail-full-exec \
	check-stage$(1)-T-$(2)-H-$(3)-rmake-exec \
	check-stage$(1)-T-$(2)-H-$(3)-rustdocck-exec \
        check-stage$(1)-T-$(2)-H-$(3)-crates-exec \
        check-stage$(1)-T-$(2)-H-$(3)-doc-crates-exec \
	check-stage$(1)-T-$(2)-H-$(3)-bench-exec \
	check-stage$(1)-T-$(2)-H-$(3)-debuginfo-gdb-exec \
	check-stage$(1)-T-$(2)-H-$(3)-debuginfo-lldb-exec \
	check-stage$(1)-T-$(2)-H-$(3)-codegen-exec \
	check-stage$(1)-T-$(2)-H-$(3)-doc-exec \
	check-stage$(1)-T-$(2)-H-$(3)-pretty-exec

# Only test the compiler-dependent crates when the target is
# able to build a compiler (when the target triple is in the set of host triples)
ifneq ($$(findstring $(2),$$(CFG_HOST)),)

check-stage$(1)-T-$(2)-H-$(3)-crates-exec: \
	$$(foreach crate,$$(TEST_CRATES), \
           check-stage$(1)-T-$(2)-H-$(3)-$$(crate)-exec)

else

check-stage$(1)-T-$(2)-H-$(3)-crates-exec: \
	$$(foreach crate,$$(TEST_TARGET_CRATES), \
           check-stage$(1)-T-$(2)-H-$(3)-$$(crate)-exec)

endif

check-stage$(1)-T-$(2)-H-$(3)-doc-crates-exec: \
        $$(foreach crate,$$(TEST_DOC_CRATES), \
           check-stage$(1)-T-$(2)-H-$(3)-doc-crate-$$(crate)-exec)

check-stage$(1)-T-$(2)-H-$(3)-doc-exec: \
        $$(foreach docname,$$(DOC_NAMES), \
           check-stage$(1)-T-$(2)-H-$(3)-doc-$$(docname)-exec) \

check-stage$(1)-T-$(2)-H-$(3)-pretty-exec: \
	check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-exec \
    check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-valgrind-exec \
	check-stage$(1)-T-$(2)-H-$(3)-pretty-rpass-full-exec \
	check-stage$(1)-T-$(2)-H-$(3)-pretty-rfail-exec \
	check-stage$(1)-T-$(2)-H-$(3)-pretty-rfail-full-exec \
	check-stage$(1)-T-$(2)-H-$(3)-pretty-bench-exec \
	check-stage$(1)-T-$(2)-H-$(3)-pretty-pretty-exec

endef

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
    $(eval $(call DEF_TEST_SETS,$(stage),$(target),$(host))))))


######################################################################
# Crate testing
######################################################################

define TEST_RUNNER

# If NO_REBUILD is set then break the dependencies on everything but
# the source files so we can test crates without rebuilding any of the
# parent crates.
ifeq ($(NO_REBUILD),)
TESTDEP_$(1)_$(2)_$(3)_$(4) = $$(SREQ$(1)_T_$(2)_H_$(3)) \
			    $$(foreach crate,$$(TARGET_CRATES), \
				$$(TLIB$(1)_T_$(2)_H_$(3))/stamp.$$(crate)) \
				$$(CRATE_FULLDEPS_$(1)_T_$(2)_H_$(3)_$(4))

else
TESTDEP_$(1)_$(2)_$(3)_$(4) = $$(RSINPUTS_$(4))
endif

$(3)/stage$(1)/test/$(4)test-$(2)$$(X_$(2)): CFG_COMPILER_HOST_TRIPLE = $(2)
$(3)/stage$(1)/test/$(4)test-$(2)$$(X_$(2)): \
		$$(CRATEFILE_$(4)) \
		$$(TESTDEP_$(1)_$(2)_$(3)_$(4))
	@$$(call E, rustc: $$@)
	$(Q)CFG_LLVM_LINKAGE_FILE=$$(LLVM_LINKAGE_PATH_$(2)) \
	    $$(subst @,,$$(STAGE$(1)_T_$(2)_H_$(3))) -o $$@ $$< --test \
		-L "$$(RT_OUTPUT_DIR_$(2))" \
		$$(LLVM_LIBDIR_RUSTFLAGS_$(2)) \
		$$(RUSTFLAGS_$(4)) \
		$$(STDCPP_LIBDIR_RUSTFLAGS_$(2))

endef

$(foreach host,$(CFG_HOST), \
 $(eval $(foreach target,$(CFG_TARGET), \
  $(eval $(foreach stage,$(STAGES), \
   $(eval $(foreach crate,$(TEST_CRATES), \
    $(eval $(call TEST_RUNNER,$(stage),$(target),$(host),$(crate))))))))))

define DEF_TEST_CRATE_RULES
check-stage$(1)-T-$(2)-H-$(3)-$(4)-exec: $$(call TEST_OK_FILE,$(1),$(2),$(3),$(4))

$$(call TEST_OK_FILE,$(1),$(2),$(3),$(4)): \
		$(3)/stage$(1)/test/$(4)test-$(2)$$(X_$(2))
	@$$(call E, run: $$<)
	$$(Q)touch $$@.start_time
	$$(Q)$$(call CFG_RUN_TEST_$(2),$$<,$(1),$(2),$(3)) $$(TESTARGS) \
	    --logfile $$(call TEST_LOG_FILE,$(1),$(2),$(3),$(4)) \
	    $$(call CRATE_TEST_EXTRA_ARGS,$(1),$(2),$(3),$(4)) \
	    && touch -r $$@.start_time $$@ && rm $$@.start_time
endef

define DEF_TEST_CRATE_RULES_android
check-stage$(1)-T-$(2)-H-$(3)-$(4)-exec: $$(call TEST_OK_FILE,$(1),$(2),$(3),$(4))

$$(call TEST_OK_FILE,$(1),$(2),$(3),$(4)): \
		$(3)/stage$(1)/test/$(4)test-$(2)$$(X_$(2))
	@$$(call E, run: $$< via adb)
	$$(Q)touch $$@.start_time
	$$(Q)$(CFG_ADB) push $$< $(CFG_ADB_TEST_DIR)
	$$(Q)$(CFG_ADB) shell '(cd $(CFG_ADB_TEST_DIR); LD_LIBRARY_PATH=./$(2) \
		./$$(notdir $$<) \
		--logfile $(CFG_ADB_TEST_DIR)/check-stage$(1)-T-$(2)-H-$(3)-$(4).log \
		$$(call CRATE_TEST_EXTRA_ARGS,$(1),$(2),$(3),$(4)) $(TESTARGS))' \
		> tmp/check-stage$(1)-T-$(2)-H-$(3)-$(4).tmp
	$$(Q)cat tmp/check-stage$(1)-T-$(2)-H-$(3)-$(4).tmp
	$$(Q)touch tmp/check-stage$(1)-T-$(2)-H-$(3)-$(4).log
	$$(Q)$(CFG_ADB) pull $(CFG_ADB_TEST_DIR)/check-stage$(1)-T-$(2)-H-$(3)-$(4).log tmp/
	$$(Q)$(CFG_ADB) shell rm $(CFG_ADB_TEST_DIR)/check-stage$(1)-T-$(2)-H-$(3)-$(4).log
	@if grep -q "result: ok" tmp/check-stage$(1)-T-$(2)-H-$(3)-$(4).tmp; \
	then \
		rm tmp/check-stage$(1)-T-$(2)-H-$(3)-$(4).tmp; \
		touch -r $$@.start_time $$@ && rm $$@.start_time; \
	else \
		rm tmp/check-stage$(1)-T-$(2)-H-$(3)-$(4).tmp; \
		exit 101; \
	fi
endef

define DEF_TEST_CRATE_RULES_null
check-stage$(1)-T-$(2)-H-$(3)-$(4)-exec: $$(call TEST_OK_FILE,$(1),$(2),$(3),$(4))

$$(call TEST_OK_FILE,$(1),$(2),$(3),$(4)): \
		$(3)/stage$(1)/test/$(4)test-$(2)$$(X_$(2))
	@$$(call E, failing: no device for $$< )
	false
endef

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
   $(foreach crate, $(TEST_CRATES), \
    $(if $(findstring $(target),$(CFG_BUILD)), \
     $(eval $(call DEF_TEST_CRATE_RULES,$(stage),$(target),$(host),$(crate))), \
     $(if $(findstring android, $(target)), \
      $(if $(findstring $(CFG_ADB_DEVICE_STATUS),"true"), \
       $(eval $(call DEF_TEST_CRATE_RULES_android,$(stage),$(target),$(host),$(crate))), \
       $(eval $(call DEF_TEST_CRATE_RULES_null,$(stage),$(target),$(host),$(crate))) \
      ), \
      $(eval $(call DEF_TEST_CRATE_RULES,$(stage),$(target),$(host),$(crate))) \
     ))))))

######################################################################
# Rules for the compiletest tests (rpass, rfail, etc.)
######################################################################

RPASS_RS := $(wildcard $(S)src/test/run-pass/*.rs)
RPASS_VALGRIND_RS := $(wildcard $(S)src/test/run-pass-valgrind/*.rs)
RPASS_FULL_RS := $(wildcard $(S)src/test/run-pass-fulldeps/*.rs)
RFAIL_FULL_RS := $(wildcard $(S)src/test/run-fail-fulldeps/*.rs)
CFAIL_FULL_RS := $(wildcard $(S)src/test/compile-fail-fulldeps/*.rs)
RFAIL_RS := $(wildcard $(S)src/test/run-fail/*.rs)
CFAIL_RS := $(wildcard $(S)src/test/compile-fail/*.rs)
PFAIL_RS := $(wildcard $(S)src/test/parse-fail/*.rs)
BENCH_RS := $(wildcard $(S)src/test/bench/*.rs)
PRETTY_RS := $(wildcard $(S)src/test/pretty/*.rs)
DEBUGINFO_GDB_RS := $(wildcard $(S)src/test/debuginfo/*.rs)
DEBUGINFO_LLDB_RS := $(wildcard $(S)src/test/debuginfo/*.rs)
CODEGEN_RS := $(wildcard $(S)src/test/codegen/*.rs)
CODEGEN_CC := $(wildcard $(S)src/test/codegen/*.cc)
RUSTDOCCK_RS := $(wildcard $(S)src/test/rustdoc/*.rs)

# perf tests are the same as bench tests only they run under
# a performance monitor.
PERF_RS := $(wildcard $(S)src/test/bench/*.rs)

RPASS_TESTS := $(RPASS_RS)
RPASS_VALGRIND_TESTS := $(RPASS_VALGRIND_RS)
RPASS_FULL_TESTS := $(RPASS_FULL_RS)
RFAIL_FULL_TESTS := $(RFAIL_FULL_RS)
CFAIL_FULL_TESTS := $(CFAIL_FULL_RS)
RFAIL_TESTS := $(RFAIL_RS)
CFAIL_TESTS := $(CFAIL_RS)
PFAIL_TESTS := $(PFAIL_RS)
BENCH_TESTS := $(BENCH_RS)
PERF_TESTS := $(PERF_RS)
PRETTY_TESTS := $(PRETTY_RS)
DEBUGINFO_GDB_TESTS := $(DEBUGINFO_GDB_RS)
DEBUGINFO_LLDB_TESTS := $(DEBUGINFO_LLDB_RS)
CODEGEN_TESTS := $(CODEGEN_RS) $(CODEGEN_CC)
RUSTDOCCK_TESTS := $(RUSTDOCCK_RS)

CTEST_SRC_BASE_rpass = run-pass
CTEST_BUILD_BASE_rpass = run-pass
CTEST_MODE_rpass = run-pass
CTEST_RUNTOOL_rpass = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_rpass-valgrind = run-pass-valgrind
CTEST_BUILD_BASE_rpass-valgrind = run-pass-valgrind
CTEST_MODE_rpass-valgrind = run-pass-valgrind
CTEST_RUNTOOL_rpass-valgrind = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_rpass-full = run-pass-fulldeps
CTEST_BUILD_BASE_rpass-full = run-pass-fulldeps
CTEST_MODE_rpass-full = run-pass
CTEST_RUNTOOL_rpass-full = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_rfail-full = run-fail-fulldeps
CTEST_BUILD_BASE_rfail-full = run-fail-fulldeps
CTEST_MODE_rfail-full = run-fail
CTEST_RUNTOOL_rfail-full = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_cfail-full = compile-fail-fulldeps
CTEST_BUILD_BASE_cfail-full = compile-fail-fulldeps
CTEST_MODE_cfail-full = compile-fail
CTEST_RUNTOOL_cfail-full = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_rfail = run-fail
CTEST_BUILD_BASE_rfail = run-fail
CTEST_MODE_rfail = run-fail
CTEST_RUNTOOL_rfail = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_cfail = compile-fail
CTEST_BUILD_BASE_cfail = compile-fail
CTEST_MODE_cfail = compile-fail
CTEST_RUNTOOL_cfail = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_pfail = parse-fail
CTEST_BUILD_BASE_pfail = parse-fail
CTEST_MODE_pfail = parse-fail
CTEST_RUNTOOL_pfail = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_bench = bench
CTEST_BUILD_BASE_bench = bench
CTEST_MODE_bench = run-pass
CTEST_RUNTOOL_bench = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_perf = bench
CTEST_BUILD_BASE_perf = perf
CTEST_MODE_perf = run-pass
CTEST_RUNTOOL_perf = $(CTEST_PERF_RUNTOOL)

CTEST_SRC_BASE_debuginfo-gdb = debuginfo
CTEST_BUILD_BASE_debuginfo-gdb = debuginfo-gdb
CTEST_MODE_debuginfo-gdb = debuginfo-gdb
CTEST_RUNTOOL_debuginfo-gdb = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_debuginfo-lldb = debuginfo
CTEST_BUILD_BASE_debuginfo-lldb = debuginfo-lldb
CTEST_MODE_debuginfo-lldb = debuginfo-lldb
CTEST_RUNTOOL_debuginfo-lldb = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_codegen = codegen
CTEST_BUILD_BASE_codegen = codegen
CTEST_MODE_codegen = codegen
CTEST_RUNTOOL_codegen = $(CTEST_RUNTOOL)

CTEST_SRC_BASE_rustdocck = rustdoc
CTEST_BUILD_BASE_rustdocck = rustdoc
CTEST_MODE_rustdocck = rustdoc
CTEST_RUNTOOL_rustdocck = $(CTEST_RUNTOOL)

# CTEST_DISABLE_$(TEST_GROUP), if set, will cause the test group to be
# disabled and the associated message to be printed as a warning
# during attempts to run those tests.

ifeq ($(CFG_GDB),)
CTEST_DISABLE_debuginfo-gdb = "no gdb found"
endif

ifeq ($(CFG_LLDB),)
CTEST_DISABLE_debuginfo-lldb = "no lldb found"
endif

ifneq ($(CFG_OSTYPE),apple-darwin)
CTEST_DISABLE_debuginfo-lldb = "lldb tests are only run on darwin"
endif

ifeq ($(CFG_OSTYPE),apple-darwin)
CTEST_DISABLE_debuginfo-gdb = "gdb on darwin needs root"
endif

ifeq ($(findstring android, $(CFG_TARGET)), android)
CTEST_DISABLE_debuginfo-gdb =
CTEST_DISABLE_debuginfo-lldb = "lldb tests are disabled on android"
endif

ifeq ($(findstring msvc,$(CFG_TARGET)),msvc)
CTEST_DISABLE_debuginfo-gdb = "gdb tests are disabled on MSVC"
endif

# CTEST_DISABLE_NONSELFHOST_$(TEST_GROUP), if set, will cause that
# test group to be disabled *unless* the target is able to build a
# compiler (i.e. when the target triple is in the set of of host
# triples).  The associated message will be printed as a warning
# during attempts to run those tests.

define DEF_CTEST_VARS

# All the per-stage build rules you might want to call from the
# command line.
#
# $(1) is the stage number
# $(2) is the target triple to test
# $(3) is the host triple to test

# Prerequisites for compiletest tests
TEST_SREQ$(1)_T_$(2)_H_$(3) = \
	$$(HBIN$(1)_H_$(3))/compiletest$$(X_$(3)) \
	$$(SREQ$(1)_T_$(2)_H_$(3))

# Rules for the cfail/rfail/rpass/bench/perf test runner

# The tests select when to use debug configuration on their own;
# remove directive, if present, from CFG_RUSTC_FLAGS (issue #7898).
CTEST_RUSTC_FLAGS := $$(subst -C debug-assertions,,$$(subst -C debug-assertions=on,,$$(CFG_RUSTC_FLAGS)))

# The tests cannot be optimized while the rest of the compiler is optimized, so
# filter out the optimization (if any) from rustc and then figure out if we need
# to be optimized
CTEST_RUSTC_FLAGS := $$(subst -O,,$$(CTEST_RUSTC_FLAGS))
ifndef CFG_DISABLE_OPTIMIZE_TESTS
CTEST_RUSTC_FLAGS += -O
endif

# Analogously to the above, whether to pass `-g` when compiling tests
# is a separate choice from whether to pass `-g` when building the
# compiler and standard library themselves.
CTEST_RUSTC_FLAGS := $$(subst -g,,$$(CTEST_RUSTC_FLAGS))
ifdef CFG_ENABLE_DEBUGINFO_TESTS
CTEST_RUSTC_FLAGS += -g
endif

CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3) := \
		--compile-lib-path $$(HLIB$(1)_H_$(3)) \
        --run-lib-path $$(TLIB$(1)_T_$(2)_H_$(3)) \
        --rustc-path $$(HBIN$(1)_H_$(3))/rustc$$(X_$(3)) \
        --rustdoc-path $$(HBIN$(1)_H_$(3))/rustdoc$$(X_$(3)) \
        --llvm-bin-path $(CFG_LLVM_INST_DIR_$(CFG_BUILD))/bin \
        --aux-base $$(S)src/test/auxiliary/ \
        --stage-id stage$(1)-$(2) \
        --target $(2) \
        --host $(3) \
	--python $$(CFG_PYTHON) \
        --gdb-version="$(CFG_GDB_VERSION)" \
        --lldb-version="$(CFG_LLDB_VERSION)" \
        --android-cross-path=$(CFG_ANDROID_CROSS_PATH) \
        --adb-path=$(CFG_ADB) \
        --adb-test-dir=$(CFG_ADB_TEST_DIR) \
        --host-rustcflags "$(RUSTC_FLAGS_$(3)) $$(CTEST_RUSTC_FLAGS) -L $$(RT_OUTPUT_DIR_$(3)) $$(STDCPP_LIBDIR_RUSTFLAGS_$(3))" \
        --lldb-python-dir=$(CFG_LLDB_PYTHON_DIR) \
        --target-rustcflags "$(RUSTC_FLAGS_$(2)) $$(CTEST_RUSTC_FLAGS) -L $$(RT_OUTPUT_DIR_$(2)) $$(STDCPP_LIBDIR_RUSTFLAGS_$(2))" \
        $$(CTEST_TESTARGS)

ifdef CFG_VALGRIND_RPASS
ifdef GOOD_VALGRIND_$(2)
CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3) += --valgrind-path "$(CFG_VALGRIND_RPASS)"
endif
endif

ifndef CFG_DISABLE_VALGRIND_RPASS
ifdef GOOD_VALGRIND_$(2)
CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3) += --force-valgrind
endif
endif

CTEST_DEPS_rpass_$(1)-T-$(2)-H-$(3) = $$(RPASS_TESTS)
CTEST_DEPS_rpass-valgrind_$(1)-T-$(2)-H-$(3) = $$(RPASS_VALGRIND_TESTS)
CTEST_DEPS_rpass-full_$(1)-T-$(2)-H-$(3) = $$(RPASS_FULL_TESTS) $$(CSREQ$(1)_T_$(3)_H_$(3)) $$(SREQ$(1)_T_$(2)_H_$(3))
CTEST_DEPS_rfail-full_$(1)-T-$(2)-H-$(3) = $$(RFAIL_FULL_TESTS) $$(CSREQ$(1)_T_$(3)_H_$(3)) $$(SREQ$(1)_T_$(2)_H_$(3))
CTEST_DEPS_cfail-full_$(1)-T-$(2)-H-$(3) = $$(CFAIL_FULL_TESTS) $$(CSREQ$(1)_T_$(3)_H_$(3)) $$(SREQ$(1)_T_$(2)_H_$(3))
CTEST_DEPS_rfail_$(1)-T-$(2)-H-$(3) = $$(RFAIL_TESTS)
CTEST_DEPS_cfail_$(1)-T-$(2)-H-$(3) = $$(CFAIL_TESTS)
CTEST_DEPS_pfail_$(1)-T-$(2)-H-$(3) = $$(PFAIL_TESTS)
CTEST_DEPS_bench_$(1)-T-$(2)-H-$(3) = $$(BENCH_TESTS)
CTEST_DEPS_perf_$(1)-T-$(2)-H-$(3) = $$(PERF_TESTS)
CTEST_DEPS_debuginfo-gdb_$(1)-T-$(2)-H-$(3) = $$(DEBUGINFO_GDB_TESTS)
CTEST_DEPS_debuginfo-lldb_$(1)-T-$(2)-H-$(3) = $$(DEBUGINFO_LLDB_TESTS) \
                                               $(S)src/etc/lldb_batchmode.py \
                                               $(S)src/etc/lldb_rust_formatters.py
CTEST_DEPS_codegen_$(1)-T-$(2)-H-$(3) = $$(CODEGEN_TESTS)
CTEST_DEPS_rustdocck_$(1)-T-$(2)-H-$(3) = $$(RUSTDOCCK_TESTS) \
        $$(HBIN$(1)_H_$(3))/rustdoc$$(X_$(3)) \
	$(S)src/etc/htmldocck.py

endef

$(foreach host,$(CFG_HOST), \
 $(eval $(foreach target,$(CFG_TARGET), \
  $(eval $(foreach stage,$(STAGES), \
   $(eval $(call DEF_CTEST_VARS,$(stage),$(target),$(host))))))))

define DEF_RUN_COMPILETEST

CTEST_ARGS$(1)-T-$(2)-H-$(3)-$(4) := \
        $$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3)) \
        --src-base $$(S)src/test/$$(CTEST_SRC_BASE_$(4))/ \
        --build-base $(3)/test/$$(CTEST_BUILD_BASE_$(4))/ \
        --mode $$(CTEST_MODE_$(4)) \
	$$(CTEST_RUNTOOL_$(4))

check-stage$(1)-T-$(2)-H-$(3)-$(4)-exec: $$(call TEST_OK_FILE,$(1),$(2),$(3),$(4))

# CTEST_DONT_RUN_$(1)-T-$(2)-H-$(3)-$(4)
# Goal: leave this variable as empty string if we should run the test.
# Otherwise, set it to the reason we are not running the test.
# (Encoded as a separate variable because GNU make does not have a
# good way to express OR on ifeq commands)

ifneq ($$(CTEST_DISABLE_$(4)),)
# Test suite is disabled for all configured targets.
CTEST_DONT_RUN_$(1)-T-$(2)-H-$(3)-$(4) := $$(CTEST_DISABLE_$(4))
else
# else, check if non-self-hosted target (i.e. target not-in hosts) ...
ifeq ($$(findstring $(2),$$(CFG_HOST)),)
# ... if so, then check if this test suite is disabled for non-selfhosts.
ifneq ($$(CTEST_DISABLE_NONSELFHOST_$(4)),)
# Test suite is disabled for this target.
CTEST_DONT_RUN_$(1)-T-$(2)-H-$(3)-$(4) := $$(CTEST_DISABLE_NONSELFHOST_$(4))
endif
endif
# Neither DISABLE nor DISABLE_NONSELFHOST is set ==> okay, run the test.
endif

ifeq ($$(CTEST_DONT_RUN_$(1)-T-$(2)-H-$(3)-$(4)),)
$$(call TEST_OK_FILE,$(1),$(2),$(3),$(4)): \
		$$(TEST_SREQ$(1)_T_$(2)_H_$(3)) \
                $$(CTEST_DEPS_$(4)_$(1)-T-$(2)-H-$(3))
	@$$(call E, run $(4) [$(2)]: $$<)
	$$(Q)touch $$@.start_time
	$$(Q)$$(call CFG_RUN_CTEST_$(2),$(1),$$<,$(3)) \
		$$(CTEST_ARGS$(1)-T-$(2)-H-$(3)-$(4)) \
		--logfile $$(call TEST_LOG_FILE,$(1),$(2),$(3),$(4)) \
                && touch -r $$@.start_time $$@ && rm $$@.start_time

else

$$(call TEST_OK_FILE,$(1),$(2),$(3),$(4)):
	@$$(call E, run $(4) [$(2)]: $$<)
	@$$(call E, warning: tests disabled: $$(CTEST_DONT_RUN_$(1)-T-$(2)-H-$(3)-$(4)))
	touch $$@

endif

endef

CTEST_NAMES = rpass rpass-valgrind rpass-full rfail-full cfail-full rfail cfail pfail \
	bench perf debuginfo-gdb debuginfo-lldb codegen rustdocck

$(foreach host,$(CFG_HOST), \
 $(eval $(foreach target,$(CFG_TARGET), \
  $(eval $(foreach stage,$(STAGES), \
   $(eval $(foreach name,$(CTEST_NAMES), \
   $(eval $(call DEF_RUN_COMPILETEST,$(stage),$(target),$(host),$(name))))))))))

PRETTY_NAMES = pretty-rpass pretty-rpass-valgrind pretty-rpass-full pretty-rfail-full pretty-rfail \
    pretty-bench pretty-pretty
PRETTY_DEPS_pretty-rpass = $(RPASS_TESTS)
PRETTY_DEPS_pretty-rpass-valgrind = $(RPASS_VALGRIND_TESTS)
PRETTY_DEPS_pretty-rpass-full = $(RPASS_FULL_TESTS)
PRETTY_DEPS_pretty-rfail-full = $(RFAIL_FULL_TESTS)
PRETTY_DEPS_pretty-rfail = $(RFAIL_TESTS)
PRETTY_DEPS_pretty-bench = $(BENCH_TESTS)
PRETTY_DEPS_pretty-pretty = $(PRETTY_TESTS)
PRETTY_DIRNAME_pretty-rpass = run-pass
PRETTY_DIRNAME_pretty-rpass-valgrind = run-pass-valgrind
PRETTY_DIRNAME_pretty-rpass-full = run-pass-fulldeps
PRETTY_DIRNAME_pretty-rfail-full = run-fail-fulldeps
PRETTY_DIRNAME_pretty-rfail = run-fail
PRETTY_DIRNAME_pretty-bench = bench
PRETTY_DIRNAME_pretty-pretty = pretty

define DEF_PRETTY_FULLDEPS
PRETTY_DEPS$(1)_T_$(2)_H_$(3)_pretty-rpass-full = $$(CSREQ$(1)_T_$(3)_H_$(3))
PRETTY_DEPS$(1)_T_$(2)_H_$(3)_pretty-rfail-full = $$(CSREQ$(1)_T_$(3)_H_$(3))
endef

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
   $(eval $(call DEF_PRETTY_FULLDEPS,$(stage),$(target),$(host))))))

define DEF_RUN_PRETTY_TEST

PRETTY_ARGS$(1)-T-$(2)-H-$(3)-$(4) := \
		$$(CTEST_COMMON_ARGS$(1)-T-$(2)-H-$(3)) \
        --src-base $$(S)src/test/$$(PRETTY_DIRNAME_$(4))/ \
        --build-base $(3)/test/$$(PRETTY_DIRNAME_$(4))/ \
        --mode pretty

check-stage$(1)-T-$(2)-H-$(3)-$(4)-exec: $$(call TEST_OK_FILE,$(1),$(2),$(3),$(4))

$$(call TEST_OK_FILE,$(1),$(2),$(3),$(4)): \
	        $$(TEST_SREQ$(1)_T_$(2)_H_$(3)) \
	        $$(PRETTY_DEPS_$(4)) \
	        $$(PRETTY_DEPS$(1)_T_$(2)_H_$(3)_$(4))
	@$$(call E, run pretty-rpass [$(2)]: $$<)
	$$(Q)touch $$@.start_time
	$$(Q)$$(call CFG_RUN_CTEST_$(2),$(1),$$<,$(3)) \
		$$(PRETTY_ARGS$(1)-T-$(2)-H-$(3)-$(4)) \
		--logfile $$(call TEST_LOG_FILE,$(1),$(2),$(3),$(4)) \
                && touch -r $$@.start_time $$@ && rm $$@.start_time

endef

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
   $(foreach pretty-name,$(PRETTY_NAMES), \
    $(eval $(call DEF_RUN_PRETTY_TEST,$(stage),$(target),$(host),$(pretty-name)))))))


######################################################################
# Crate & freestanding documentation tests
######################################################################

define DEF_RUSTDOC
RUSTDOC_EXE_$(1)_T_$(2)_H_$(3) := $$(HBIN$(1)_H_$(3))/rustdoc$$(X_$(3))
RUSTDOC_$(1)_T_$(2)_H_$(3) := $$(RPATH_VAR$(1)_T_$(2)_H_$(3)) $$(RUSTDOC_EXE_$(1)_T_$(2)_H_$(3))
endef

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
   $(eval $(call DEF_RUSTDOC,$(stage),$(target),$(host))))))

# Freestanding

define DEF_DOC_TEST

check-stage$(1)-T-$(2)-H-$(3)-doc-$(4)-exec: $$(call TEST_OK_FILE,$(1),$(2),$(3),doc-$(4))

# If NO_REBUILD is set then break the dependencies on everything but
# the source files so we can test documentation without rebuilding
# rustdoc etc.
ifeq ($(NO_REBUILD),)
DOCTESTDEP_$(1)_$(2)_$(3)_$(4) = \
	$$(DOCFILE_$(4)) \
	$$(TEST_SREQ$(1)_T_$(2)_H_$(3)) \
	$$(RUSTDOC_EXE_$(1)_T_$(2)_H_$(3))
else
DOCTESTDEP_$(1)_$(2)_$(3)_$(4) = $$(DOCFILE_$(4))
endif

ifeq ($(2),$$(CFG_BUILD))
$$(call TEST_OK_FILE,$(1),$(2),$(3),doc-$(4)): $$(DOCTESTDEP_$(1)_$(2)_$(3)_$(4))
	@$$(call E, run doc-$(4) [$(2)])
	$$(Q)touch $$@.start_time
	$$(Q)$$(RUSTDOC_$(1)_T_$(2)_H_$(3)) --cfg dox --test $$< \
		--test-args "$$(TESTARGS)" && \
		touch -r $$@.start_time $$@ && rm $$@.start_time
else
$$(call TEST_OK_FILE,$(1),$(2),$(3),doc-$(4)):
	touch $$@
endif
endef

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
   $(foreach docname,$(DOC_NAMES), \
    $(eval $(call DEF_DOC_TEST,$(stage),$(target),$(host),$(docname)))))))

# Crates

define DEF_CRATE_DOC_TEST

# If NO_REBUILD is set then break the dependencies on everything but
# the source files so we can test crate documentation without
# rebuilding any of the parent crates.
ifeq ($(NO_REBUILD),)
CRATEDOCTESTDEP_$(1)_$(2)_$(3)_$(4) = \
	$$(TEST_SREQ$(1)_T_$(2)_H_$(3)) \
	$$(CRATE_FULLDEPS_$(1)_T_$(2)_H_$(3)_$(4)) \
	$$(RUSTDOC_EXE_$(1)_T_$(2)_H_$(3))
else
CRATEDOCTESTDEP_$(1)_$(2)_$(3)_$(4) = $$(RSINPUTS_$(4))
endif

check-stage$(1)-T-$(2)-H-$(3)-doc-crate-$(4)-exec: \
	$$(call TEST_OK_FILE,$(1),$(2),$(3),doc-crate-$(4))

ifeq ($(2),$$(CFG_BUILD))
$$(call TEST_OK_FILE,$(1),$(2),$(3),doc-crate-$(4)): $$(CRATEDOCTESTDEP_$(1)_$(2)_$(3)_$(4))
	@$$(call E, run doc-crate-$(4) [$(2)])
	$$(Q)touch $$@.start_time
	$$(Q)CFG_LLVM_LINKAGE_FILE=$$(LLVM_LINKAGE_PATH_$(2)) \
	    $$(RUSTDOC_$(1)_T_$(2)_H_$(3)) --test --cfg dox \
	        $$(CRATEFILE_$(4)) --test-args "$$(TESTARGS)" && \
	        touch -r $$@.start_time $$@ && rm $$@.start_time
else
$$(call TEST_OK_FILE,$(1),$(2),$(3),doc-crate-$(4)):
	touch $$@
endif

endef

$(foreach host,$(CFG_HOST), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach stage,$(STAGES), \
   $(foreach crate,$(TEST_DOC_CRATES), \
    $(eval $(call DEF_CRATE_DOC_TEST,$(stage),$(target),$(host),$(crate)))))))

######################################################################
# Shortcut rules
######################################################################

TEST_GROUPS = \
	crates \
	$(foreach crate,$(TEST_CRATES),$(crate)) \
	$(foreach crate,$(TEST_DOC_CRATES),doc-crate-$(crate)) \
	rpass \
    rpass-valgrind \
	rpass-full \
	rfail-full \
	cfail-full \
	rfail \
	cfail \
	pfail \
	bench \
	perf \
	rmake \
	rustdocck \
	debuginfo-gdb \
	debuginfo-lldb \
	codegen \
	doc \
	$(foreach docname,$(DOC_NAMES),doc-$(docname)) \
	pretty \
	pretty-rpass \
    pretty-rpass-valgrind \
	pretty-rpass-full \
	pretty-rfail-full \
	pretty-rfail \
	pretty-bench \
	pretty-pretty \
	$(NULL)

define DEF_CHECK_FOR_STAGE_AND_TARGET_AND_HOST
check-stage$(1)-T-$(2)-H-$(3): check-stage$(1)-T-$(2)-H-$(3)-exec
endef

$(foreach stage,$(STAGES), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach host,$(CFG_HOST), \
   $(eval $(call DEF_CHECK_FOR_STAGE_AND_TARGET_AND_HOST,$(stage),$(target),$(host))))))

define DEF_CHECK_FOR_STAGE_AND_TARGET_AND_HOST_AND_GROUP
check-stage$(1)-T-$(2)-H-$(3)-$(4): check-stage$(1)-T-$(2)-H-$(3)-$(4)-exec
endef

$(foreach stage,$(STAGES), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach host,$(CFG_HOST), \
   $(foreach group,$(TEST_GROUPS), \
    $(eval $(call DEF_CHECK_FOR_STAGE_AND_TARGET_AND_HOST_AND_GROUP,$(stage),$(target),$(host),$(group)))))))

define DEF_CHECK_FOR_STAGE
check-stage$(1): check-stage$(1)-H-$$(CFG_BUILD)
check-stage$(1)-H-all: $$(foreach target,$$(CFG_TARGET), \
                           check-stage$(1)-H-$$(target))
endef

$(foreach stage,$(STAGES), \
 $(eval $(call DEF_CHECK_FOR_STAGE,$(stage))))

define DEF_CHECK_FOR_STAGE_AND_GROUP
check-stage$(1)-$(2): check-stage$(1)-H-$$(CFG_BUILD)-$(2)
check-stage$(1)-H-all-$(2): $$(foreach target,$$(CFG_TARGET), \
                               check-stage$(1)-H-$$(target)-$(2))
endef

$(foreach stage,$(STAGES), \
 $(foreach group,$(TEST_GROUPS), \
  $(eval $(call DEF_CHECK_FOR_STAGE_AND_GROUP,$(stage),$(group)))))


define DEF_CHECK_FOR_STAGE_AND_HOSTS
check-stage$(1)-H-$(2): $$(foreach target,$$(CFG_TARGET), \
                           check-stage$(1)-T-$$(target)-H-$(2))
endef

$(foreach stage,$(STAGES), \
 $(foreach host,$(CFG_HOST), \
  $(eval $(call DEF_CHECK_FOR_STAGE_AND_HOSTS,$(stage),$(host)))))

define DEF_CHECK_FOR_STAGE_AND_HOSTS_AND_GROUP
check-stage$(1)-H-$(2)-$(3): $$(foreach target,$$(CFG_TARGET), \
                                check-stage$(1)-T-$$(target)-H-$(2)-$(3))
endef

$(foreach stage,$(STAGES), \
 $(foreach host,$(CFG_HOST), \
  $(foreach group,$(TEST_GROUPS), \
   $(eval $(call DEF_CHECK_FOR_STAGE_AND_HOSTS_AND_GROUP,$(stage),$(host),$(group))))))

define DEF_CHECK_DOC_FOR_STAGE
check-stage$(1)-docs: $$(foreach docname,$$(DOC_NAMES), \
                       check-stage$(1)-T-$$(CFG_BUILD)-H-$$(CFG_BUILD)-doc-$$(docname)) \
                     $$(foreach crate,$$(TEST_DOC_CRATES), \
                       check-stage$(1)-T-$$(CFG_BUILD)-H-$$(CFG_BUILD)-doc-crate-$$(crate))
endef

$(foreach stage,$(STAGES), \
 $(eval $(call DEF_CHECK_DOC_FOR_STAGE,$(stage))))

define DEF_CHECK_CRATE
check-$(1): check-stage2-T-$$(CFG_BUILD)-H-$$(CFG_BUILD)-$(1)-exec
endef

$(foreach crate,$(TEST_CRATES), \
 $(eval $(call DEF_CHECK_CRATE,$(crate))))

######################################################################
# RMAKE rules
######################################################################

RMAKE_TESTS := $(shell ls -d $(S)src/test/run-make/*/)
RMAKE_TESTS := $(RMAKE_TESTS:$(S)src/test/run-make/%/=%)

define DEF_RMAKE_FOR_T_H
# $(1) the stage
# $(2) target triple
# $(3) host triple


ifeq ($(2)$(3),$$(CFG_BUILD)$$(CFG_BUILD))
check-stage$(1)-T-$(2)-H-$(3)-rmake-exec: \
		$$(call TEST_OK_FILE,$(1),$(2),$(3),rmake)

$$(call TEST_OK_FILE,$(1),$(2),$(3),rmake): \
		$$(RMAKE_TESTS:%=$(3)/test/run-make/%-$(1)-T-$(2)-H-$(3).ok)
	@touch $$@

$(3)/test/run-make/%-$(1)-T-$(2)-H-$(3).ok: \
	export INCLUDE := $$(CFG_MSVC_INCLUDE_PATH_$$(HOST_$(3)))
$(3)/test/run-make/%-$(1)-T-$(2)-H-$(3).ok: \
	export LIB := $$(CFG_MSVC_LIB_PATH_$$(HOST_$(3)))
$(3)/test/run-make/%-$(1)-T-$(2)-H-$(3).ok: \
		$(S)src/test/run-make/%/Makefile \
		$$(CSREQ$(1)_T_$(2)_H_$(3))
	@rm -rf $(3)/test/run-make/$$*
	@mkdir -p $(3)/test/run-make/$$*
	$$(Q)touch $$@.start_time
	$$(Q)$$(CFG_PYTHON) $(S)src/etc/maketest.py $$(dir $$<) \
        $$(MAKE) \
	    $$(HBIN$(1)_H_$(3))/rustc$$(X_$(3)) \
	    $(3)/test/run-make/$$* \
	    '$$(CC_$(3))' \
	    "$$(CFG_GCCISH_CFLAGS_$(3))" \
	    $$(HBIN$(1)_H_$(3))/rustdoc$$(X_$(3)) \
	    "$$(TESTNAME)" \
	    $$(LD_LIBRARY_PATH_ENV_NAME$(1)_T_$(2)_H_$(3)) \
	    "$$(LD_LIBRARY_PATH_ENV_HOSTDIR$(1)_T_$(2)_H_$(3))" \
	    "$$(LD_LIBRARY_PATH_ENV_TARGETDIR$(1)_T_$(2)_H_$(3))" \
	    $(1) \
	    $$(S) \
	    $(3)
	@touch -r $$@.start_time $$@ && rm $$@.start_time
else
# FIXME #11094 - The above rule doesn't work right for multiple targets
check-stage$(1)-T-$(2)-H-$(3)-rmake-exec:
	@true

endif


endef

$(foreach stage,$(STAGES), \
 $(foreach target,$(CFG_TARGET), \
  $(foreach host,$(CFG_HOST), \
   $(eval $(call DEF_RMAKE_FOR_T_H,$(stage),$(target),$(host))))))
