# needs-profiler-support
# ignore-windows-gnu

# FIXME(mati865): MinGW GCC miscompiles compiler-rt profiling library but with Clang it works
# properly. Since we only have GCC on the CI ignore the test for now.

-include ../coverage/coverage_tools.mk

BASEDIR=../coverage-reports
SOURCEDIR=../coverage

# The `llvm-cov show` flag `--debug`, used to generate the `counters` output files, is only enabled
# if LLVM assertions are enabled. Requires Rust config `llvm/optimize` and not
# `llvm/release_debuginfo`. Note that some CI builds disable debug assertions (by setting
# `NO_LLVM_ASSERTIONS=1`), so it is not OK to fail the test, but `bless`ed test results cannot be
# generated without debug assertions.
LLVM_COV_DEBUG := $(shell \
		"$(LLVM_BIN_DIR)"/llvm-cov show --debug 2>&1 | \
		grep -q "Unknown command line argument '--debug'"; \
		echo $$?)
ifeq ($(LLVM_COV_DEBUG), 1)
DEBUG_FLAG=--debug
endif

# FIXME(richkadel): I'm adding `--ignore-filename-regex=` line(s) for specific test(s) that produce
# `llvm-cov` results for multiple files (for example `uses_crate.rs` and `used_crate/mod.rs`) as a
# workaround for two problems causing tests to fail on Windows:
#
# 1. When multiple files appear in the `llvm-cov show` results, each file's coverage results can
#    appear in different a different order. Whether this is random or, somehow, platform-specific,
#    the Windows output flips the order of the files, compared to Linux. In the `uses_crate.rs`
#    test, the only test-unique (interesting) results we care about are the results for only one
#    of the two files, `mod/uses_crate.rs`, so the workaround is to ignore all but this one file.
#    In the future, we may want a more sophisticated solution that splits apart `llvm-cov show`
#    results into separate results files for each result (taking care not to create new file
#    paths that might be too long for Windows MAX_PATH limits when creating these new sub-results,
#    as well).
# 2. When multiple files appear in the `llvm-cov show` results, the results for each file are
#    prefixed with their filename, including platform-specific path separators (`\` for Windows,
#    and `/` everywhere else). This could be filtered or normalized of course, but by ignoring
#    coverage results for all but one of the file, the filenames are no longer included anyway.
#    If this changes (if/when we decide to support `llvm-cov show` results for multiple files),
#    the file path separator differences may need to be addressed.
#
# Since this is only a workaround, I decided to implement the override by adding an option for
# each file to be ignored, using a `--ignore-filename-regex=` entry for each one, rather than
# implement some more sophisticated solution with a new custom test directive in the test file
# itself (similar to `expect-exit-status`) because that would add a lot of complexity and still
# be a workaround, with the same result, with no benefit.
#
# Yes these `--ignore-filename-regex=` options are included in all invocations of `llvm-cov show`
# for now, but it is effectively ignored for all tests that don't include this file anyway.
#
# Note that it's also possible the `_counters.<test>.txt` and `<test>.json` files may order
# results from multiple files inconsistently, which might also have to be accomodated if and when
# we allow `llvm-cov` to produce results for multiple files. (The path separators appear to be
# normalized to `/` in those files, thankfully.) But since we are ignoring results for all but one
# file, this workaround addresses those potential issues as well.
LLVM_COV_IGNORE_FILES=\
	--ignore-filename-regex=uses_crate.rs

# When generating `expected_*` results (using `x.py test --bless`), the `--debug` flag is forced.
# If assertions are disabled, the command will fail with an error, rather than attempt to generate
# only partial results.
ifdef RUSTC_BLESS_TEST
DEBUG_FLAG=--debug
endif

ifeq ($(LLVM_VERSION_11_PLUS),true)
all: $(patsubst $(SOURCEDIR)/lib/%.rs,%,$(wildcard $(SOURCEDIR)/lib/*.rs)) $(patsubst $(SOURCEDIR)/%.rs,%,$(wildcard $(SOURCEDIR)/*.rs))
else
$(info Rust option `-Z instrument-coverage` requires LLVM 11 or higher. Test skipped.)
all:
endif

# Ensure there are no `expected` results for tests that may have been removed or renamed
.PHONY: clear_expected_if_blessed
clear_expected_if_blessed:
ifdef RUSTC_BLESS_TEST
	rm -f expected_export_coverage.*.json
	rm -f expected_show_coverage.*.txt
	rm -f expected_show_coverage_counters.*.txt
endif

-include clear_expected_if_blessed

%: $(SOURCEDIR)/lib/%.rs
	# Compile the test library with coverage instrumentation
	$(RUSTC) $(SOURCEDIR)/lib/$@.rs \
			$$( grep -q '^\/\/ require-rust-edition-2018' $(SOURCEDIR)/lib/$@.rs && echo "--edition=2018" ) \
			--crate-type rlib -Zinstrument-coverage

%: $(SOURCEDIR)/%.rs
	# Compile the test program with coverage instrumentation
	$(RUSTC) $(SOURCEDIR)/$@.rs \
			$$( grep -q '^\/\/ require-rust-edition-2018' $(SOURCEDIR)/$@.rs && echo "--edition=2018" ) \
			-L "$(TMPDIR)" -Zinstrument-coverage

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
			$(LLVM_COV_IGNORE_FILES) \
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
	#
	# FIXME(richkadel): It looks like most past variations seem to have been mitigated. None of the
	# Rust test source samples have the `// ignore-llvm-cov-show-diffs` anymore. The main variation
	# I had seen (and is still present in the new `coverage/lib/used_crate.rs`) is the `llvm-cov show`
	# reporting of multiple instantiations of a generic function with different type substitutions.
	# For some reason, `llvm-cov show` can report these in a non-deterministic order, breaking the
	# `diff` comparision. I was able to work around the problem with `diff --ignore-matching-lines=RE`
	# to ignore each line prefixing each generic instantiation coverage code region.

	$(DIFF) --ignore-matching-lines='::<.*>.*:$$' \
		expected_show_coverage.$@.txt "$(TMPDIR)"/actual_show_coverage.$@.txt || \
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
			$(LLVM_COV_IGNORE_FILES) \
			--summary-only \
			--instr-profile="$(TMPDIR)"/$@.profdata \
			$(call BIN,"$(TMPDIR)"/$@) \
		| "$(PYTHON)" $(BASEDIR)/prettify_json.py \
		> "$(TMPDIR)"/actual_export_coverage.$@.json
	# FIXME(richkadel): With the addition of `--ignore-matching-lines=RE` to ignore the
	# non-deterministically-ordered coverage results for multiple instantiations of generics with
	# differing type substitutions, I probably don't need the `.json` files anymore (and may not
	# need `prettify_json.py` either).

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/actual_export_coverage.$@.json expected_export_coverage.$@.json
else
	# Check that exported JSON coverage data matches what we expect (`--bless` refreshes `expected`)
	$(DIFF) expected_export_coverage.$@.json "$(TMPDIR)"/actual_export_coverage.$@.json
endif
