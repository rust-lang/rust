-include ../tools.mk

all:
	cp foo.rs $(TMPDIR)/.foo.rs
	$(RUSTC) $(TMPDIR)/.foo.rs 2>&1 \
		| grep "invalid character.*in crate name:"
	cp foo.rs $(TMPDIR)/.foo.bar
	$(RUSTC) $(TMPDIR)/.foo.bar 2>&1 \
		| grep "invalid character.*in crate name:"
	cp foo.rs $(TMPDIR)/+foo+bar
	$(RUSTC) $(TMPDIR)/+foo+bar 2>&1 \
		| grep "invalid character.*in crate name:"
