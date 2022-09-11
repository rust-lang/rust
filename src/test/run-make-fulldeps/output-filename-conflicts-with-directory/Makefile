include ../tools.mk

all:
	cp foo.rs $(TMPDIR)/foo.rs
	mkdir $(TMPDIR)/foo
	$(RUSTC) $(TMPDIR)/foo.rs -o $(TMPDIR)/foo 2>&1 \
		| $(CGREP) -e "the generated executable for the input file \".*foo\.rs\" conflicts with the existing directory \".*foo\""
