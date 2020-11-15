# needs-profiler-support
# ignore-windows-gnu

# FIXME(mati865): MinGW GCC miscompiles compiler-rt profiling library but with Clang it works
# properly. Since we only have GCC on the CI ignore the test for now.

# ISSUE(76038): When targeting MSVC, Rust binaries built with both `-Z instrument-coverage` and
# `-C link-dead-code` typically crash (with a seg-fault) or at best generate an empty `*.profraw`.
# See ../coverage/coverage_tools.mk for more information.

-include ../coverage/coverage_tools.mk

BASEDIR=../coverage-reports-base
SOURCEDIR=../coverage

# The `llvm-cov show` flag `--debug`, used to generate the `counters` output files, is only enabled
# if LLVM assertions are enabled. Requires Rust config `llvm/optimize` and not
# `llvm/release_debuginfo`. Note that some CI builds disable debug assertions (by setting
# `NO_LLVM_ASSERTIONS=1`), so it is not OK to fail the test, but `bless`ed test results cannot be
# generated without debug assertions.
LLVM_COV_DEBUG := $(shell "$(LLVM_BIN_DIR)"/llvm-cov show --debug 2>&1 | grep -q "Unknown command line argument '--debug'"; echo $$?)
ifeq ($(LLVM_COV_DEBUG), 1)
DEBUG_FLAG=--debug
endif

# When generating `expected_*` results (using `x.py test --bless`), the `--debug` flag is forced.
# If assertions are disabled, the command will fail with an error, rather than attempt to generate
# only partial results.
ifdef RUSTC_BLESS_TEST
DEBUG_FLAG=--debug
endif

all: $(patsubst $(SOURCEDIR)/%.rs,%,$(wildcard $(SOURCEDIR)/*.rs))

# Ensure there are no `expected` results for tests that may have been removed or renamed
.PHONY: clear_expected_if_blessed
clear_expected_if_blessed:
ifdef RUSTC_BLESS_TEST
	rm -f expected_export_coverage.*.json
	rm -f expected_show_coverage.*.txt
	rm -f expected_show_coverage_counters.*.txt
endif

-include clear_expected_if_blessed

%: $(SOURCEDIR)/%.rs
	# Compile the test program with coverage instrumentation and generate relevant MIR.
	$(RUSTC) $(SOURCEDIR)/$@.rs \
			-Zinstrument-coverage \
			-Clink-dead-code=$(LINK_DEAD_CODE)

	# Run it in order to generate some profiling data,
	# with `LLVM_PROFILE_FILE=<profdata_file>` environment variable set to
	# output the coverage stats for this run.
	LLVM_PROFILE_FILE="$(TMPDIR)"/$@.profraw \
			$(call RUN,$@) || \
			( \
				status=$$?; \
				grep -q "^\/\/ expect-exit-status-$$status" $(SOURCEDIR)/$@.rs || \
				( >&2 echo "program exited with an unexpected exit status: $$status"; \
					false \
				) \
			)

	# Postprocess the profiling data so it can be used by the llvm-cov tool
	"$(LLVM_BIN_DIR)"/llvm-profdata merge --sparse \
			"$(TMPDIR)"/$@.profraw \
			-o "$(TMPDIR)"/$@.profdata

	# Generate a coverage report using `llvm-cov show`.
	"$(LLVM_BIN_DIR)"/llvm-cov show \
			$(DEBUG_FLAG) \
			--Xdemangler="$(RUST_DEMANGLER)" \
			--show-line-counts-or-regions \
			--instr-profile="$(TMPDIR)"/$@.profdata \
			$(call BIN,"$(TMPDIR)"/$@) \
		> "$(TMPDIR)"/actual_show_coverage.$@.txt \
		2> "$(TMPDIR)"/show_coverage_stderr.$@.txt || \
	( status=$$? ; \
		>&2 cat "$(TMPDIR)"/show_coverage_stderr.$@.txt ; \
		exit $$status \
	)

ifdef DEBUG_FLAG
	# The first line (beginning with "Args:" contains hard-coded, build-specific
	# file paths. Strip that line and keep the remaining lines with counter debug
	# data.
	tail -n +2 "$(TMPDIR)"/show_coverage_stderr.$@.txt \
		> "$(TMPDIR)"/actual_show_coverage_counters.$@.txt
endif

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/actual_show_coverage.$@.txt \
			expected_show_coverage.$@.txt
	cp "$(TMPDIR)"/actual_show_coverage_counters.$@.txt \
			expected_show_coverage_counters.$@.txt
else
	# Compare the show coverage output (`--bless` refreshes `typical` files)
	# Note `llvm-cov show` output for some programs can vary, but can be ignored
	# by inserting `// ignore-llvm-cov-show-diffs` at the top of the source file.

	$(DIFF) expected_show_coverage.$@.txt "$(TMPDIR)"/actual_show_coverage.$@.txt || \
		( grep -q '^\/\/ ignore-llvm-cov-show-diffs' $(SOURCEDIR)/$@.rs && \
			>&2 echo 'diff failed, but suppressed with `// ignore-llvm-cov-show-diffs` in $(SOURCEDIR)/$@.rs' \
		) || \
		( >&2 echo 'diff failed, and not suppressed without `// ignore-llvm-cov-show-diffs` in $(SOURCEDIR)/$@.rs'; \
			false \
		)

ifdef DEBUG_FLAG
	$(DIFF) expected_show_coverage_counters.$@.txt "$(TMPDIR)"/actual_show_coverage_counters.$@.txt || \
		( grep -q '^\/\/ ignore-llvm-cov-show-diffs' $(SOURCEDIR)/$@.rs && \
			>&2 echo 'diff failed, but suppressed with `// ignore-llvm-cov-show-diffs` in $(SOURCEDIR)/$@.rs' \
		) || \
		( >&2 echo 'diff failed, and not suppressed without `// ignore-llvm-cov-show-diffs` in $(SOURCEDIR)/$@.rs'; \
			>&2 echo '(Ignore anyway until mangled function names in "counters" files are demangled.)' \
		)

		# FIXME(richkadel): Apply the demangler to the `*_show_coverage_counters.*.txt` files,
		# so the crate disambiguator differences will be stripped away. At that point, these files
		# will be less likely to vary, and the last `echo` above (starting with "Ignore anyway")
		# can be replaced with `false` to fail the test.
endif

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
