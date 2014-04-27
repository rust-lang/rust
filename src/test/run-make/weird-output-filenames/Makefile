-include ../tools.mk

all:
	$(RUSTC) foo.rs -o $(TMPDIR)/.foo
	rm $(TMPDIR)/.foo
	$(RUSTC) foo.rs -o $(TMPDIR)/.foo.bar
	rm $(TMPDIR)/.foo.bar
	$(RUSTC) foo.rs -o $(TMPDIR)/+foo+bar
	rm $(TMPDIR)/$(call BIN,+foo+bar)
