# needs-profiler-support
# ignore-windows-gnu

-include ../../run-make-fulldeps/tools.mk

# FIXME(eddyb) provide `HOST_RUSTC` and `TARGET_RUSTC`
# instead of hardcoding them everywhere they're needed.
ifeq ($(IS_MUSL_HOST),1)
ADDITIONAL_ARGS := $(RUSTFLAGS)
endif

all:
	# Generate PGO profiles
	$(BARE_RUSTC) $(ADDITIONAL_ARGS) -Cprofile-generate=$(TMPDIR)/profiles --out-dir $(TMPDIR) main.rs
	$(TMPDIR)/main

	# Merge profiles
	"$(LLVM_BIN_DIR)/llvm-profdata" merge \
		-o "$(TMPDIR)/merged.profdata" \
		"$(TMPDIR)/profiles" || exit 1

	# Use the profile
	$(RUSTC) -Cprofile-use=$(TMPDIR)/merged.profdata --emit dep-info main.rs

	# Check that profile file is in depinfo
	$(CGREP) "merged.profdata" < $(TMPDIR)/main.d
