# needs-profiler-support
# ignore-windows-gnu

# Rust coverage maps support LLVM Coverage Mapping Format versions 5 and 6,
# corresponding with LLVM versions 12 and 13, respectively.
# When upgrading LLVM versions, consider whether to enforce a minimum LLVM
# version during testing, with an additional directive at the top of this file
# that sets, for example: `min-llvm-version: 12.0`

# FIXME(mati865): MinGW GCC miscompiles compiler-rt profiling library but with Clang it works
# properly. Since we only have GCC on the CI ignore the test for now.

-include ../coverage/coverage_tools.mk

BASEDIR=../coverage-reports
SOURCEDIR=../coverage

# The `llvm-cov show` flag `--debug`, used to generate the `counters` output files, is only
# enabled if LLVM assertions are enabled. This requires Rust config `llvm/optimize` and not
# `llvm/release_debuginfo`. Note that some CI builds disable debug assertions (by setting
# `NO_LLVM_ASSERTIONS=1`), so the tests must still pass even if the `--debug` flag is
# not supported. (Note that `counters` files are only produced in the `$(TMPDIR)`
# directory, for inspection and debugging support. They are *not* copied to `expected_*`
# files when `--bless`ed.)
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
# (Note that it's also possible the `_counters.<test>.txt` and `<test>.json` files (if generated)
# may order results from multiple files inconsistently, which might also have to be accomodated
# if and when we allow `llvm-cov` to produce results for multiple files. Note, the path separators
# appear to be normalized to `/` in those files, thankfully.)
LLVM_COV_IGNORE_FILES=\
	--ignore-filename-regex='(uses_crate.rs|uses_inline_crate.rs|unused_mod.rs)'

all: $(patsubst $(SOURCEDIR)/lib/%.rs,%,$(wildcard $(SOURCEDIR)/lib/*.rs)) $(patsubst $(SOURCEDIR)/%.rs,%,$(wildcard $(SOURCEDIR)/*.rs))

# Ensure there are no `expected` results for tests that may have been removed or renamed
.PHONY: clear_expected_if_blessed
clear_expected_if_blessed:
ifdef RUSTC_BLESS_TEST
	rm -f expected_*
endif

-include clear_expected_if_blessed

%: $(SOURCEDIR)/lib/%.rs
	# Compile the test library with coverage instrumentation
	$(RUSTC) $(SOURCEDIR)/lib/$@.rs \
			$$( sed -n 's/^\/\/ compile-flags: \([^#]*\).*/\1/p' $(SOURCEDIR)/lib/$@.rs ) \
			--crate-type rlib -Cinstrument-coverage

%: $(SOURCEDIR)/%.rs
	# Compile the test program with coverage instrumentation
	$(RUSTC) $(SOURCEDIR)/$@.rs \
			$$( sed -n 's/^\/\/ compile-flags: \([^#]*\).*/\1/p' $(SOURCEDIR)/$@.rs ) \
			-L "$(TMPDIR)" -Cinstrument-coverage

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

	# Run it through rustdoc as well to cover doctests.
	# `%p` is the pid, and `%m` the binary signature. We suspect that the pid alone
	# might result in overwritten files and failed tests, as rustdoc spawns each
	# doctest as its own process, so make sure the filename is as unique as possible.
	LLVM_PROFILE_FILE="$(TMPDIR)"/$@-%p-%m.profraw \
			$(RUSTDOC) --crate-name workaround_for_79771 --test $(SOURCEDIR)/$@.rs \
			$$( sed -n 's/^\/\/ compile-flags: \([^#]*\).*/\1/p' $(SOURCEDIR)/$@.rs ) \
			-L "$(TMPDIR)" -Cinstrument-coverage \
			-Z unstable-options --persist-doctests=$(TMPDIR)/rustdoc-$@

	# Postprocess the profiling data so it can be used by the llvm-cov tool
	"$(LLVM_BIN_DIR)"/llvm-profdata merge --sparse \
			"$(TMPDIR)"/$@*.profraw \
			-o "$(TMPDIR)"/$@.profdata

	# Generate a coverage report using `llvm-cov show`.
	"$(LLVM_BIN_DIR)"/llvm-cov show \
			$(DEBUG_FLAG) \
			$(LLVM_COV_IGNORE_FILES) \
			--compilation-dir=. \
			--Xdemangler="$(RUST_DEMANGLER)" \
			--show-line-counts-or-regions \
			--instr-profile="$(TMPDIR)"/$@.profdata \
			$(call BIN,"$(TMPDIR)"/$@) \
			$$( \
				for file in $(TMPDIR)/rustdoc-$@/*/rust_out; do \
				[ -x "$$file" ] && printf "%s %s " -object $$file; \
				done \
			) \
		2> "$(TMPDIR)"/show_coverage_stderr.$@.txt \
		| "$(PYTHON)" $(BASEDIR)/normalize_paths.py \
		> "$(TMPDIR)"/actual_show_coverage.$@.txt || \
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
else
	# Compare the show coverage output (`--bless` refreshes `typical` files).
	#
	# FIXME(richkadel): None of the Rust test source samples have the
	# `// ignore-llvm-cov-show-diffs` anymore. This directive exists to work around a limitation
	# with `llvm-cov show`. When reporting coverage for multiple instantiations of a generic function,
	# with different type substitutions, `llvm-cov show` prints these in a non-deterministic order,
	# breaking the `diff` comparision.
	#
	# A partial workaround is implemented below, with `diff --ignore-matching-lines=RE`
	# to ignore each line prefixing each generic instantiation coverage code region.
	#
	# This workaround only works if the coverage counts are identical across all reported
	# instantiations. If there is no way to ensure this, you may need to apply the
	# `// ignore-llvm-cov-show-diffs` directive, and check for differences using the
	# `.json` files to validate that results have not changed. (Until then, the JSON
	# files are redundant, so there is no need to generate `expected_*.json` files or
	# compare actual JSON results.)

	$(DIFF) --ignore-matching-lines='^  | .*::<.*>.*:$$' --ignore-matching-lines='^  | <.*>::.*:$$' \
		expected_show_coverage.$@.txt "$(TMPDIR)"/actual_show_coverage.$@.txt || \
		( grep -q '^\/\/ ignore-llvm-cov-show-diffs' $(SOURCEDIR)/$@.rs && \
			>&2 echo 'diff failed, but suppressed with `// ignore-llvm-cov-show-diffs` in $(SOURCEDIR)/$@.rs' \
		) || \
		( >&2 echo 'diff failed, and not suppressed without `// ignore-llvm-cov-show-diffs` in $(SOURCEDIR)/$@.rs'; \
			false \
		)
endif
