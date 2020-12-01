# needs-profiler-support

# ISSUE(76038): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`.
# See ../coverage/coverage_tools.mk for more information.

-include ../coverage/coverage_tools.mk

BASEDIR=../coverage-llvmir-base

ifeq ($(UNAME),Darwin)
	INSTR_PROF_DATA_SUFFIX=,regular,live_support
	DATA_SECTION_PREFIX=__DATA,
	LLVM_COV_SECTION_PREFIX=__LLVM_COV,
	COMDAT_IF_SUPPORTED=
else
	INSTR_PROF_DATA_SUFFIX=
	DATA_SECTION_PREFIX=
	LLVM_COV_SECTION_PREFIX=
	COMDAT_IF_SUPPORTED=, comdat
endif

ifeq ($(LINK_DEAD_CODE),yes)
	DEFINE_INTERNAL=define hidden
else
	DEFINE_INTERNAL=define internal
endif

ifdef IS_WINDOWS
	LLVM_FILECHECK_OPTIONS=\
		-check-prefixes=CHECK,WINDOWS \
		-DPRIVATE_GLOBAL='internal global' \
		-DDEFINE_INTERNAL='$(DEFINE_INTERNAL)' \
		-DCOMDAT_IF_SUPPORTED='$(COMDAT_IF_SUPPORTED)' \
		-DINSTR_PROF_DATA='.lprfd$$M' \
		-DINSTR_PROF_NAME='.lprfn$$M' \
		-DINSTR_PROF_CNTS='.lprfc$$M' \
		-DINSTR_PROF_VALS='.lprfv$$M' \
		-DINSTR_PROF_VNODES='.lprfnd$$M' \
		-DINSTR_PROF_COVMAP='.lcovmap$$M' \
		-DINSTR_PROF_COVFUN='.lcovfun$$M' \
		-DINSTR_PROF_ORDERFILE='.lorderfile$$M'
else
	LLVM_FILECHECK_OPTIONS=\
		-check-prefixes=CHECK \
		-DPRIVATE_GLOBAL='private global' \
		-DDEFINE_INTERNAL='$(DEFINE_INTERNAL)' \
		-DCOMDAT_IF_SUPPORTED='$(COMDAT_IF_SUPPORTED)' \
		-DINSTR_PROF_DATA='$(DATA_SECTION_PREFIX)__llvm_prf_data$(INSTR_PROF_DATA_SUFFIX)' \
		-DINSTR_PROF_NAME='$(DATA_SECTION_PREFIX)__llvm_prf_names' \
		-DINSTR_PROF_CNTS='$(DATA_SECTION_PREFIX)__llvm_prf_cnts' \
		-DINSTR_PROF_VALS='$(DATA_SECTION_PREFIX)__llvm_prf_vals' \
		-DINSTR_PROF_VNODES='$(DATA_SECTION_PREFIX)__llvm_prf_vnds' \
		-DINSTR_PROF_COVMAP='$(LLVM_COV_SECTION_PREFIX)__llvm_covmap' \
		-DINSTR_PROF_COVFUN='$(LLVM_COV_SECTION_PREFIX)__llvm_covfun' \
		-DINSTR_PROF_ORDERFILE='$(DATA_SECTION_PREFIX)__llvm_orderfile'
endif

ifeq ($(LLVM_VERSION_11_PLUS),true)
all: test_llvm_ir
else
$(info Rust option `-Z instrument-coverage` requires LLVM 11 or higher. Test skipped.)
all:
endif

test_llvm_ir:
	# Compile the test program with non-experimental coverage instrumentation, and generate LLVM IR
	#
	# Note: `-Clink-dead-code=no` disables the option, needed because the option is automatically
	# enabled for some platforms, but not for Windows MSVC (due to Issue #76038). The state of this
	# option affects the generated MIR and coverage, so it is enabled for tests to ensure the
	# tests results are the same across platforms.
	$(RUSTC) $(BASEDIR)/testprog.rs \
			-Zinstrument-coverage \
			-Clink-dead-code=$(LINK_DEAD_CODE) \
			--emit=llvm-ir

	cat "$(TMPDIR)"/testprog.ll | \
			"$(LLVM_FILECHECK)" $(BASEDIR)/filecheck.testprog.txt $(LLVM_FILECHECK_OPTIONS)
