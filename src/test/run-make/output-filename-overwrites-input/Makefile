-include ../tools.mk

all:
	cp foo.rs $(TMPDIR)/foo
	$(RUSTC) $(TMPDIR)/foo -o $(TMPDIR)/foo 2>&1 \
		| $(CGREP) -e "the input file \".*foo\" would be overwritten by the generated executable"
	cp bar.rs $(TMPDIR)/bar.rlib
	$(RUSTC) $(TMPDIR)/bar.rlib -o $(TMPDIR)/bar.rlib 2>&1 \
		| $(CGREP) -e "the input file \".*bar.rlib\" would be overwritten by the generated executable"
	$(RUSTC) foo.rs 2>&1 && $(RUSTC) -Z ls $(TMPDIR)/foo 2>&1
	cp foo.rs $(TMPDIR)/foo.rs
	$(RUSTC) $(TMPDIR)/foo.rs -o $(TMPDIR)/foo.rs 2>&1 \
		| $(CGREP) -e "the input file \".*foo.rs\" would be overwritten by the generated executable"
