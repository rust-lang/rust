-include ../../run-make-fulldeps/tools.mk

# FIXME(eddyb) provide `HOST_RUSTC` and `TARGET_RUSTC`
# instead of hardcoding them everywhere they're needed.
all:
	$(BARE_RUSTC) foo.rs --out-dir $(TMPDIR)
	$(RUSTC) bar.rs --target $(TARGET) --emit dep-info
	$(CGREP) -v "proc-macro source" < $(TMPDIR)/bar.d
