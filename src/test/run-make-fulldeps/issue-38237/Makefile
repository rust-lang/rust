include ../tools.mk

all:
	$(RUSTC) foo.rs; $(RUSTC) bar.rs
	$(RUSTDOC) baz.rs -L $(TMPDIR) -o $(TMPDIR)
