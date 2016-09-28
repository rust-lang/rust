-include ../tools.mk

all:
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs --emit dep-info
	grep "rustc-macro source" $(TMPDIR)/bar.d && exit 1 || exit 0
