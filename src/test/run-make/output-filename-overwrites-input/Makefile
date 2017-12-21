-include ../tools.mk

all:
	cp foo.rs $(TMPDIR)/foo
	$(RUSTC) $(TMPDIR)/foo 2>&1 \
		| $(CGREP) -e "the input file \".*foo\" would be overwritten by the generated executable"
	$(RUSTC) foo.rs 2>&1 && $(RUSTC) -Z ls $(TMPDIR)/foo 2>&1
	cp foo.rs $(TMPDIR)/foo.rs
	$(RUSTC) $(TMPDIR)/foo.rs -o $(TMPDIR)/foo.rs 2>&1 \
		| $(CGREP) -e "the input file \".*foo.rs\" would be overwritten by the generated executable"
