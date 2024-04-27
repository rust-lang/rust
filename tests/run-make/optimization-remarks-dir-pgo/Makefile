# needs-profiler-support
# ignore-windows-gnu
# ignore-cross-compile

# FIXME(mati865): MinGW GCC miscompiles compiler-rt profiling library but with Clang it works
# properly. Since we only have GCC on the CI ignore the test for now.

include ../tools.mk

PROFILE_DIR=$(TMPDIR)/profiles

check_hotness:
	$(RUSTC) -Cprofile-generate="$(TMPDIR)"/profdata -O foo.rs -o$(TMPDIR)/foo
	$(TMPDIR)/foo
	"$(LLVM_BIN_DIR)"/llvm-profdata merge \
		-o "$(TMPDIR)"/merged.profdata \
		"$(TMPDIR)"/profdata/*.profraw
	$(RUSTC) -Cprofile-use=$(TMPDIR)/merged.profdata -O foo.rs -Cremark=all -Zremark-dir=$(PROFILE_DIR)

	# Check that PGO hotness is included in the remark files
	cat $(PROFILE_DIR)/*.opt.yaml | $(CGREP) -e "Hotness"
