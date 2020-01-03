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

# For some very small programs GNU ld seems to not properly handle
# instrumentation sections correctly. Neither Gold nor LLD have that problem.
ifeq ($(UNAME),Linux)
ifneq (,$(findstring x86,$(TARGET)))
COMMON_FLAGS=-Clink-args=-fuse-ld=gold
endif
endif


all:
	# We don't compile `opaque` with either optimizations or instrumentation.
	$(RUSTC) $(COMMON_FLAGS) opaque.rs || exit 1
	# Compile the test program with instrumentation
	mkdir -p "$(TMPDIR)/prof_data_dir" || exit 1
	$(RUSTC) $(COMMON_FLAGS) interesting.rs \
		-Cprofile-generate="$(TMPDIR)/prof_data_dir" -O -Ccodegen-units=1 || exit 1
	$(RUSTC) $(COMMON_FLAGS) main.rs -Cprofile-generate="$(TMPDIR)/prof_data_dir" -O || exit 1
	# The argument below generates to the expected branch weights
	$(call RUN,main aaaaaaaaaaaa2bbbbbbbbbbbb2bbbbbbbbbbbbbbbbcc) || exit 1
	"$(LLVM_BIN_DIR)/llvm-profdata" merge \
		-o "$(TMPDIR)/prof_data_dir/merged.profdata" \
		"$(TMPDIR)/prof_data_dir" || exit 1
	$(RUSTC) $(COMMON_FLAGS) interesting.rs \
		-Cprofile-use="$(TMPDIR)/prof_data_dir/merged.profdata" -O \
		-Ccodegen-units=1 --emit=llvm-ir || exit 1
	cat "$(TMPDIR)/interesting.ll" | "$(LLVM_FILECHECK)" filecheck-patterns.txt
