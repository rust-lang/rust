-include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs --emit dep-info
	$(CGREP) -v "proc-macro source" < $(TMPDIR)/bar.d
