# needs-profiler-support
# ignore-msvc

# FIXME(richkadel): Debug the following problem, and reenable on Windows (by
# removing the `# ignore-msvc` directive above). The current implementation
# generates a segfault when running the instrumented `testprog` executable,
# after the `main()` function completes, but before the process terminates.
# This most likely points to a problem generating the LLVM "testprog.profraw"
# file.

-include ../tools.mk

UNAME = $(shell uname)

ifeq ($(UNAME),Darwin)
	INSTR_PROF_DATA_SUFFIX=,regular,live_support
	DATA_SECTION_PREFIX=__DATA,
	LLVM_COV_SECTION_PREFIX=__LLVM_COV,
else
	INSTR_PROF_DATA_SUFFIX=
	DATA_SECTION_PREFIX=
	LLVM_COV_SECTION_PREFIX=
endif

# This test makes sure that LLVM coverage maps are genereated in LLVM IR.

COMMON_FLAGS=-Zinstrument-coverage

all:
	# Compile the test program with instrumentation, and also generate LLVM IR
	$(RUSTC) $(COMMON_FLAGS) testprog.rs \
			--emit=link,llvm-ir

	# check the LLVM IR
ifdef IS_WIN32
	cat "$(TMPDIR)"/testprog.ll | "$(LLVM_FILECHECK)" filecheck-patterns.txt \
			-check-prefixes=CHECK,WIN32 \
			-DPRIVATE_GLOBAL="internal global" \
			-DINSTR_PROF_DATA=".lprfd$$M" \
			-DINSTR_PROF_NAME=".lprfn$$M" \
			-DINSTR_PROF_CNTS=".lprfc$$M" \
			-DINSTR_PROF_VALS=".lprfv$$M" \
			-DINSTR_PROF_VNODES=".lprfnd$$M" \
			-DINSTR_PROF_COVMAP=".lcovmap$$M" \
			-DINSTR_PROF_ORDERFILE=".lorderfile$$M"
else
	cat "$(TMPDIR)"/testprog.ll | "$(LLVM_FILECHECK)" filecheck-patterns.txt \
			-check-prefixes=CHECK \
			-DPRIVATE_GLOBAL="private global" \
			-DINSTR_PROF_DATA="$(DATA_SECTION_PREFIX)__llvm_prf_data$(INSTR_PROF_DATA_SUFFIX)" \
			-DINSTR_PROF_NAME="$(DATA_SECTION_PREFIX)__llvm_prf_names" \
			-DINSTR_PROF_CNTS="$(DATA_SECTION_PREFIX)__llvm_prf_cnts" \
			-DINSTR_PROF_VALS="$(DATA_SECTION_PREFIX)__llvm_prf_vals" \
			-DINSTR_PROF_VNODES="$(DATA_SECTION_PREFIX)__llvm_prf_vnds" \
			-DINSTR_PROF_COVMAP="$(LLVM_COV_SECTION_PREFIX)__llvm_covmap" \
			-DINSTR_PROF_ORDERFILE="$(DATA_SECTION_PREFIX)__llvm_orderfile"
endif

	# Run it in order to generate some profiling data,
	# with `LLVM_PROFILE_FILE=<profdata_file>` environment variable set to
	# output the coverage stats for this run.
	LLVM_PROFILE_FILE="$(TMPDIR)"/testprog.profraw \
			$(call RUN,testprog)

	# Postprocess the profiling data so it can be used by the llvm-cov tool
	"$(LLVM_BIN_DIR)"/llvm-profdata merge --sparse \
			"$(TMPDIR)"/testprog.profraw \
			-o "$(TMPDIR)"/testprog.profdata

	# Generate a coverage report using `llvm-cov show`. The output ordering
	# can be non-deterministic, so ignore the return status. If the test fails
	# when comparing the JSON `export`, the `show` output may be useful when
	# debugging.
	"$(LLVM_BIN_DIR)"/llvm-cov show \
			--Xdemangler="$(RUST_DEMANGLER)" \
			--show-line-counts-or-regions \
			--instr-profile="$(TMPDIR)"/testprog.profdata \
			$(call BIN,"$(TMPDIR)"/testprog) \
		> "$(TMPDIR)"/actual_show_coverage.txt

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/actual_show_coverage.txt typical_show_coverage.txt
else
	# Compare the show coverage output
	$(DIFF) typical_show_coverage.txt "$(TMPDIR)"/actual_show_coverage.txt || \
		>&2 echo 'diff failed for `llvm-cov show` (might not be an error)'
endif

	# Generate a coverage report in JSON, using `llvm-cov export`, and fail if
	# there are differences from the expected output.
	"$(LLVM_BIN_DIR)"/llvm-cov export \
			--summary-only \
			--instr-profile="$(TMPDIR)"/testprog.profdata \
			$(call BIN,"$(TMPDIR)"/testprog) \
		| "$(PYTHON)" prettify_json.py \
		> "$(TMPDIR)"/actual_export_coverage.json

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/actual_export_coverage.json expected_export_coverage.json
else
	# Check that the exported JSON coverage data matches what we expect
	$(DIFF) expected_export_coverage.json "$(TMPDIR)"/actual_export_coverage.json
endif
