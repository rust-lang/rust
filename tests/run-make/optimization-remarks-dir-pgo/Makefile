# needs-profiler-support
# ignore-cross-compile

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
