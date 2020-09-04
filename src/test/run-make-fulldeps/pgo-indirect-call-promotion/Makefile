# needs-profiler-support
# ignore-windows-gnu

# FIXME(mati865): MinGW GCC miscompiles compiler-rt profiling library but with Clang it works
# properly. Since we only have GCC on the CI ignore the test for now.

-include ../tools.mk

# This test makes sure that indirect call promotion is performed. The test
# programs calls the same function a thousand times through a function pointer.
# Only PGO data provides the information that it actually always is the same
# function. We verify that the indirect call promotion pass inserts a check
# whether it can make a direct call instead of the indirect call.

# LLVM doesn't support instrumenting binaries that use SEH:
# https://github.com/rust-lang/rust/issues/61002
#
# Things work fine with -Cpanic=abort though.
ifdef IS_MSVC
COMMON_FLAGS=-Cpanic=abort
endif

all:
	# We don't compile `opaque` with either optimizations or instrumentation.
	# We don't compile `opaque` with either optimizations or instrumentation.
	$(RUSTC) $(COMMON_FLAGS) opaque.rs
	# Compile the test program with instrumentation
	mkdir -p "$(TMPDIR)"/prof_data_dir
	$(RUSTC) $(COMMON_FLAGS) interesting.rs \
		-Cprofile-generate="$(TMPDIR)"/prof_data_dir -O -Ccodegen-units=1
	$(RUSTC) $(COMMON_FLAGS) main.rs -Cprofile-generate="$(TMPDIR)"/prof_data_dir -O
	# The argument below generates to the expected branch weights
	$(call RUN,main) || exit 1
	"$(LLVM_BIN_DIR)"/llvm-profdata merge \
		-o "$(TMPDIR)"/prof_data_dir/merged.profdata \
		"$(TMPDIR)"/prof_data_dir
	$(RUSTC) $(COMMON_FLAGS) interesting.rs \
		-Cprofile-use="$(TMPDIR)"/prof_data_dir/merged.profdata -O \
		-Ccodegen-units=1 --emit=llvm-ir
	cat "$(TMPDIR)"/interesting.ll | "$(LLVM_FILECHECK)" filecheck-patterns.txt
