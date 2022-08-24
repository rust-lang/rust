include ../tools.mk

all:
	$(RUSTC) foo.rs
	rm $(TMPDIR)/$(call BIN,foo)
	$(RUSTC) foo.rs --crate-name bar
	rm $(TMPDIR)/$(call BIN,bar)
	$(RUSTC) foo1.rs
	rm $(TMPDIR)/$(call BIN,foo)
	$(RUSTC) foo1.rs -o $(TMPDIR)/$(call BIN,bar1)
	rm $(TMPDIR)/$(call BIN,bar1)
