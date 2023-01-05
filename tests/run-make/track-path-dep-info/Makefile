include ../../run-make-fulldeps/tools.mk

# FIXME(eddyb) provide `HOST_RUSTC` and `TARGET_RUSTC`
# instead of hardcoding them everywhere they're needed.
ifeq ($(IS_MUSL_HOST),1)
ADDITIONAL_ARGS := $(RUSTFLAGS)
endif

all:
	# Proc macro
	$(BARE_RUSTC) $(ADDITIONAL_ARGS) --out-dir $(TMPDIR) macro_def.rs
	EXISTING_PROC_MACRO_ENV=1 $(RUSTC) --emit dep-info macro_use.rs
	$(CGREP) "emojis.txt:" < $(TMPDIR)/macro_use.d
