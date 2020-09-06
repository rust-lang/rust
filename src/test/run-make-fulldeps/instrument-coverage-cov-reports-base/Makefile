# needs-profiler-support
# ignore-windows-gnu

# FIXME(mati865): MinGW GCC miscompiles compiler-rt profiling library but with Clang it works
# properly. Since we only have GCC on the CI ignore the test for now.

# ISSUE(76038): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`.
# See ../instrument-coverage/coverage_tools.mk for more information.

-include ../instrument-coverage/coverage_tools.mk

BASEDIR=../instrument-coverage-cov-reports-base
SOURCEDIR=../instrument-coverage

all: $(patsubst $(SOURCEDIR)/%.rs,%,$(wildcard $(SOURCEDIR)/*.rs))

# Ensure there are no `expected` results for tests that may have been removed or renamed
.PHONY: clear_expected_if_blessed
clear_expected_if_blessed:
ifdef RUSTC_BLESS_TEST
	rm -f expected_export_coverage.*.json
	rm -f typical_show_coverage.*.txt
endif

-include clear_expected_if_blessed

%: $(SOURCEDIR)/%.rs
	# Compile the test program with "experimental" coverage instrumentation and generate relevant MIR.
	#
	# FIXME(richkadel):  `-Zexperimental-coverage` to `-Zinstrument-coverage` once we are
	# satisfied with the branch-level instrumentation.
	$(RUSTC) $(SOURCEDIR)/$@.rs \
			-Zexperimental-coverage \
			-Clink-dead-code=$(LINK_DEAD_CODE)

	# Run it in order to generate some profiling data,
	# with `LLVM_PROFILE_FILE=<profdata_file>` environment variable set to
	# output the coverage stats for this run.
	LLVM_PROFILE_FILE="$(TMPDIR)"/$@.profraw \
			$(call RUN,$@)

	# Postprocess the profiling data so it can be used by the llvm-cov tool
	"$(LLVM_BIN_DIR)"/llvm-profdata merge --sparse \
			"$(TMPDIR)"/$@.profraw \
			-o "$(TMPDIR)"/$@.profdata

	# Generate a coverage report using `llvm-cov show`. The output ordering
	# can be non-deterministic, so ignore the return status. If the test fails
	# when comparing the JSON `export`, the `show` output may be useful when
	# debugging.
	"$(LLVM_BIN_DIR)"/llvm-cov show \
			--Xdemangler="$(RUST_DEMANGLER)" \
			--show-line-counts-or-regions \
			--instr-profile="$(TMPDIR)"/$@.profdata \
			$(call BIN,"$(TMPDIR)"/$@) \
		> "$(TMPDIR)"/actual_show_coverage.$@.txt

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/actual_show_coverage.$@.txt typical_show_coverage.$@.txt
else
	# Compare the show coverage output (`--bless` refreshes `typical` files)
	$(DIFF) typical_show_coverage.$@.txt "$(TMPDIR)"/actual_show_coverage.$@.txt || \
		>&2 echo 'diff failed for `llvm-cov show` on $@ (might not be an error)'
endif

	# Generate a coverage report in JSON, using `llvm-cov export`, and fail if
	# there are differences from the expected output.
	"$(LLVM_BIN_DIR)"/llvm-cov export \
			--summary-only \
			--instr-profile="$(TMPDIR)"/$@.profdata \
			$(call BIN,"$(TMPDIR)"/$@) \
		| "$(PYTHON)" $(BASEDIR)/prettify_json.py \
		> "$(TMPDIR)"/actual_export_coverage.$@.json

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/actual_export_coverage.$@.json expected_export_coverage.$@.json
else
	# Check that exported JSON coverage data matches what we expect (`--bless` refreshes `expected`)
	$(DIFF) expected_export_coverage.$@.json "$(TMPDIR)"/actual_export_coverage.$@.json
endif
