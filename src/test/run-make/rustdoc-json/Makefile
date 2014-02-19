-include ../tools.mk
all:
	$(RUSTDOC) -w json -o $(TMPDIR)/doc.json foo.rs
	$(RUSTDOC) -o $(TMPDIR)/doc $(TMPDIR)/doc.json
