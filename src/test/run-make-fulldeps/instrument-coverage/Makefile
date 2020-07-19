# needs-profiler-support
# ignore-msvc

# FIXME(richkadel): Debug the following problem, and reenable on Windows (by
# removing the `# ignore-msvc` directive above). The current implementation
# generates a segfault when running the instrumented `main` executable,
# after the `main` program code executes, but before the process terminates.
# This most likely points to a problem generating the LLVM "main.profraw"
# file.

-include ../tools.mk

# This test makes sure that LLVM coverage maps are genereated in LLVM IR.

COMMON_FLAGS=-Zinstrument-coverage

all:
	# Compile the test program with instrumentation, and also generate LLVM IR
	$(RUSTC) $(COMMON_FLAGS) main.rs

	# Run it in order to generate some profiling data,
	# with `LLVM_PROFILE_FILE=<profdata_file>` environment variable set to
	# output the coverage stats for this run.
	LLVM_PROFILE_FILE="$(TMPDIR)"/main.profraw \
	  $(call RUN,main)

	# Postprocess the profiling data so it can be used by the llvm-cov tool
	"$(LLVM_BIN_DIR)"/llvm-profdata merge --sparse \
	  "$(TMPDIR)"/main.profraw \
		-o "$(TMPDIR)"/main.profdata

	# Generate a coverage report using `llvm-cov show`. The output ordering
	# can be non-deterministic, so ignore the return status. If the test fails
	# when comparing the JSON `export`, the `show` output may be useful when
	# debugging.
	"$(LLVM_BIN_DIR)"/llvm-cov show \
	  --Xdemangler="$(RUST_DEMANGLER)" \
	  --show-line-counts-or-regions \
	  --instr-profile="$(TMPDIR)"/main.profdata \
		$(call BIN,"$(TMPDIR)"/main) \
		> "$(TMPDIR)"/actual_show_coverage.txt

	# Compare the show coverage output
	$(DIFF) typical_show_coverage.txt "$(TMPDIR)"/actual_show_coverage.txt || \
	  >&2 echo 'diff failed for `llvm-cov show` (might not be an error)'

	# Generate a coverage report in JSON, using `llvm-cov export`, and fail if
	# there are differences from the expected output.
	"$(LLVM_BIN_DIR)"/llvm-cov export \
	  --summary-only \
	  --instr-profile="$(TMPDIR)"/main.profdata \
		$(call BIN,"$(TMPDIR)"/main) \
		| "$(PYTHON)" prettify_json.py \
		> "$(TMPDIR)"/actual_export_coverage.json

	# Check that the exported JSON coverage data matches what we expect
	$(DIFF) expected_export_coverage.json "$(TMPDIR)"/actual_export_coverage.json
