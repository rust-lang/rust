-include ../tools.mk

all:
	$(RUSTC) -o $(TMPDIR)/input.dd -Z no-analysis --emit dep-info input.rs
	sed -i "s@$(TMPDIR)/@@g" $(TMPDIR)/input.dd
	diff -u $(TMPDIR)/input.dd input.dd
