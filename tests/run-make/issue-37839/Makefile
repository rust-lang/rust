include ../tools.mk

all:
	$(RUSTC) a.rs && $(RUSTC) b.rs
	$(BARE_RUSTC) c.rs -L dependency=$(TMPDIR) --extern b=$(TMPDIR)/libb.rlib \
		--out-dir=$(TMPDIR)
