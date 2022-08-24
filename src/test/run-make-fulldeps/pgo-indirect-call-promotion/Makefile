# needs-profiler-support
# ignore-windows-gnu

# FIXME(mati865): MinGW GCC miscompiles compiler-rt profiling library but with Clang it works
# properly. Since we only have GCC on the CI ignore the test for now.

include ../tools.mk

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
