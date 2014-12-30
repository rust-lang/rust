-include ../tools.mk

all: $(TMPDIR)/libnative.a
	mkdir -p $(TMPDIR)/crate
	mkdir -p $(TMPDIR)/native
	mv $(TMPDIR)/libnative.a $(TMPDIR)/native
	$(RUSTC) a.rs
	mv $(TMPDIR)/liba.rlib $(TMPDIR)/crate
	$(RUSTC) b.rs -L $(TMPDIR)/crate:native && exit 1 || exit 0
	$(RUSTC) b.rs -L $(TMPDIR)/crate:dependency && exit 1 || exit 0
	$(RUSTC) b.rs -L $(TMPDIR)/crate:crate
	$(RUSTC) b.rs -L $(TMPDIR)/crate
	$(RUSTC) c.rs -L $(TMPDIR)/crate:native && exit 1 || exit 0
	$(RUSTC) c.rs -L $(TMPDIR)/crate:crate && exit 1 || exit 0
	$(RUSTC) c.rs -L $(TMPDIR)/crate:dependency
	$(RUSTC) c.rs -L $(TMPDIR)/crate
	$(RUSTC) d.rs -L $(TMPDIR)/native:dependency && exit 1 || exit 0
	$(RUSTC) d.rs -L $(TMPDIR)/native:crate && exit 1 || exit 0
	$(RUSTC) d.rs -L $(TMPDIR)/native:native
	$(RUSTC) d.rs -L $(TMPDIR)/native
	mkdir -p $(TMPDIR)/e1
	mkdir -p $(TMPDIR)/e2
	$(RUSTC) e.rs -o $(TMPDIR)/e1/libe.rlib
	$(RUSTC) e.rs -o $(TMPDIR)/e2/libe.rlib
	$(RUSTC) f.rs -L $(TMPDIR)/e1 -L $(TMPDIR)/e2 && exit 1 || exit 0
	$(RUSTC) f.rs -L $(TMPDIR)/e1:crate -L $(TMPDIR)/e2 && exit 1 || exit 0
	$(RUSTC) f.rs -L $(TMPDIR)/e1:crate -L $(TMPDIR)/e2:crate && exit 1 || exit 0
	$(RUSTC) f.rs -L $(TMPDIR)/e1:native -L $(TMPDIR)/e2
	$(RUSTC) f.rs -L $(TMPDIR)/e1:dependency -L $(TMPDIR)/e2
	$(RUSTC) f.rs -L $(TMPDIR)/e1:dependency -L $(TMPDIR)/e2:crate
