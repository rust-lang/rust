-include ../tools.mk

all:
	$(RUSTC) -o $(TMPDIR)/input.dd -Z no-analysis --emit dep-info input.rs
	sed -i'.bak' 's/^.*input.dd/input.dd/g' $(TMPDIR)/input.dd
	diff -u $(TMPDIR)/input.dd input.dd
