-include ../tools.mk

all:
	$(RUSTC) foo.rs -o $(TMPDIR)/.foo 2>&1 \
		| grep "invalid character in crate name:"
	$(RUSTC) foo.rs -o $(TMPDIR)/.foo.bar 2>&1 \
		| grep "invalid character in crate name:"
	$(RUSTC) foo.rs -o $(TMPDIR)/+foo+bar 2>&1 \
		| grep "invalid character in crate name:"
