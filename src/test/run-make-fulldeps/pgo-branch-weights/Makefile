# needs-profiler-support

-include ../tools.mk

# This test makes sure that instrumented binaries record the right counts for
# functions being called and branches being taken. We run an instrumented binary
# with an argument that causes a know path through the program and then check
# that the expected counts get added to the use-phase LLVM IR.

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
	$(call RUN,main aaaaaaaaaaaa2bbbbbbbbbbbb2bbbbbbbbbbbbbbbbcc) || exit 1
	"$(LLVM_BIN_DIR)"/llvm-profdata merge \
		-o "$(TMPDIR)"/prof_data_dir/merged.profdata \
		"$(TMPDIR)"/prof_data_dir
	$(RUSTC) $(COMMON_FLAGS) interesting.rs \
		-Cprofile-use="$(TMPDIR)"/prof_data_dir/merged.profdata -O \
		-Ccodegen-units=1 --emit=llvm-ir
	cat "$(TMPDIR)"/interesting.ll | "$(LLVM_FILECHECK)" filecheck-patterns.txt
