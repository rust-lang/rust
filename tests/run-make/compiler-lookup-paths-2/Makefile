include ../tools.mk

all:
	mkdir -p $(TMPDIR)/a $(TMPDIR)/b
	$(RUSTC) a.rs && mv $(TMPDIR)/liba.rlib $(TMPDIR)/a
	$(RUSTC) b.rs -L $(TMPDIR)/a && mv $(TMPDIR)/libb.rlib $(TMPDIR)/b
	$(RUSTC) c.rs -L crate=$(TMPDIR)/b -L dependency=$(TMPDIR)/a \
		&& exit 1 || exit 0
