# needs-profiler-support

-include ../tools.mk

# This test makes sure that PGO profiling data leads to cold functions being
# marked as `cold` and hot functions with `inlinehint`.
# The test program contains an `if` were actual execution only ever takes the
# `else` branch. Accordingly, we expect the function that is never called to
# be marked as cold.
#
# The program is compiled with `-Copt-level=s` because this setting disables
# LLVM's pre-inlining pass (i.e. a pass that does some inlining before it adds
# the profiling instrumentation). Disabling this pass leads to rather
# predictable IR which we need for this test to be stable.

COMMON_FLAGS=-Copt-level=s -Ccodegen-units=1

# LLVM doesn't support instrumenting binaries that use SEH:
# https://github.com/rust-lang/rust/issues/61002
#
# Things work fine with -Cpanic=abort though.
ifdef IS_MSVC
COMMON_FLAGS+= -Cpanic=abort
endif

ifeq ($(UNAME),Darwin)
# macOS does not have the `tac` command, but `tail -r` does the same thing
TAC := tail -r
else
# some other platforms don't support the `-r` flag for `tail`, so use `tac`
TAC := tac
endif

all:
	# Compile the test program with instrumentation
	$(RUSTC) $(COMMON_FLAGS) -Cprofile-generate="$(TMPDIR)" main.rs
	# Run it in order to generate some profiling data
	$(call RUN,main some-argument) || exit 1
	# Postprocess the profiling data so it can be used by the compiler
	"$(LLVM_BIN_DIR)"/llvm-profdata merge \
		-o "$(TMPDIR)"/merged.profdata \
		"$(TMPDIR)"/default_*.profraw
	# Compile the test program again, making use of the profiling data
	$(RUSTC) $(COMMON_FLAGS) -Cprofile-use="$(TMPDIR)"/merged.profdata --emit=llvm-ir main.rs
	# Check that the generate IR contains some things that we expect
	#
	# We feed the file into LLVM FileCheck tool *in reverse* so that we see the
	# line with the function name before the line with the function attributes.
	# FileCheck only supports checking that something matches on the next line,
	# but not if something matches on the previous line.
	$(TAC) "$(TMPDIR)"/main.ll | "$(LLVM_FILECHECK)" filecheck-patterns.txt
