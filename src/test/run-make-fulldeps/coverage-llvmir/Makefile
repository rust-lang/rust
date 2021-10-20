# needs-profiler-support

-include ../coverage/coverage_tools.mk

BASEDIR=../coverage-llvmir

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

DEFINE_INTERNAL=define internal

ifdef IS_WINDOWS
	LLVM_FILECHECK_OPTIONS=\
		-check-prefixes=CHECK,WINDOWS \
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

all: test_llvm_ir

test_llvm_ir:
	# Compile the test program with non-experimental coverage instrumentation, and generate LLVM IR
	$(RUSTC) $(BASEDIR)/testprog.rs \
			-Zinstrument-coverage \
			--emit=llvm-ir

	cat "$(TMPDIR)"/testprog.ll | \
			"$(LLVM_FILECHECK)" $(BASEDIR)/filecheck.testprog.txt $(LLVM_FILECHECK_OPTIONS)
