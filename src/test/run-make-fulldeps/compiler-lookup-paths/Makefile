-include ../tools.mk

all: $(TMPDIR)/libnative.a
	mkdir -p $(TMPDIR)/crate
	mkdir -p $(TMPDIR)/native
	mv $(TMPDIR)/libnative.a $(TMPDIR)/native
	$(RUSTC) a.rs
	mv $(TMPDIR)/liba.rlib $(TMPDIR)/crate
	$(RUSTC) b.rs -L native=$(TMPDIR)/crate && exit 1 || exit 0
	$(RUSTC) b.rs -L dependency=$(TMPDIR)/crate && exit 1 || exit 0
	$(RUSTC) b.rs -L crate=$(TMPDIR)/crate
	$(RUSTC) b.rs -L all=$(TMPDIR)/crate
	$(RUSTC) c.rs -L native=$(TMPDIR)/crate && exit 1 || exit 0
	$(RUSTC) c.rs -L crate=$(TMPDIR)/crate && exit 1 || exit 0
	$(RUSTC) c.rs -L dependency=$(TMPDIR)/crate
	$(RUSTC) c.rs -L all=$(TMPDIR)/crate
	$(RUSTC) d.rs -L dependency=$(TMPDIR)/native && exit 1 || exit 0
	$(RUSTC) d.rs -L crate=$(TMPDIR)/native && exit 1 || exit 0
	$(RUSTC) d.rs -L native=$(TMPDIR)/native
	$(RUSTC) d.rs -L all=$(TMPDIR)/native
	# Deduplication tests:
	#   Same hash, no errors.
	mkdir -p $(TMPDIR)/e1
	mkdir -p $(TMPDIR)/e2
	$(RUSTC) e.rs -o $(TMPDIR)/e1/libe.rlib
	$(RUSTC) e.rs -o $(TMPDIR)/e2/libe.rlib
	$(RUSTC) f.rs -L $(TMPDIR)/e1 -L $(TMPDIR)/e2
	$(RUSTC) f.rs -L crate=$(TMPDIR)/e1 -L $(TMPDIR)/e2
	$(RUSTC) f.rs -L crate=$(TMPDIR)/e1 -L crate=$(TMPDIR)/e2
	#   Different hash, errors.
	$(RUSTC) e2.rs -o $(TMPDIR)/e2/libe.rlib
	$(RUSTC) f.rs -L $(TMPDIR)/e1 -L $(TMPDIR)/e2 && exit 1 || exit 0
	$(RUSTC) f.rs -L crate=$(TMPDIR)/e1 -L $(TMPDIR)/e2 && exit 1 || exit 0
	$(RUSTC) f.rs -L crate=$(TMPDIR)/e1 -L crate=$(TMPDIR)/e2 && exit 1 || exit 0
	#   Native/dependency paths don't cause errors.
	$(RUSTC) f.rs -L native=$(TMPDIR)/e1 -L $(TMPDIR)/e2
	$(RUSTC) f.rs -L dependency=$(TMPDIR)/e1 -L $(TMPDIR)/e2
	$(RUSTC) f.rs -L dependency=$(TMPDIR)/e1 -L crate=$(TMPDIR)/e2
