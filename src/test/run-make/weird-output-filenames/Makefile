-include ../tools.mk

all:
	cp foo.rs $(TMPDIR)/.foo.rs
	$(RUSTC) $(TMPDIR)/.foo.rs 2>&1 \
		| $(CGREP) -e "invalid character.*in crate name:"
	cp foo.rs $(TMPDIR)/.foo.bar
	$(RUSTC) $(TMPDIR)/.foo.bar 2>&1 \
		| $(CGREP) -e "invalid character.*in crate name:"
	cp foo.rs $(TMPDIR)/+foo+bar.rs
	$(RUSTC) $(TMPDIR)/+foo+bar.rs 2>&1 \
		| $(CGREP) -e "invalid character.*in crate name:"
	cp foo.rs $(TMPDIR)/-foo.rs
	$(RUSTC) $(TMPDIR)/-foo.rs 2>&1 \
		| $(CGREP) 'crate names cannot start with a `-`'
