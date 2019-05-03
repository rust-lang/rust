-include ../tools.mk

# ignore-windows

NAME := $(shell $(RUSTC) --print file-names foo.rs)

all:
	mkdir -p $(TMPDIR)/outdir
	$(RUSTC) foo.rs -o $(TMPDIR)/outdir/$(NAME)
	ln -nsf outdir/$(NAME) $(TMPDIR)
	RUSTC_LOG=rustc_metadata::loader $(RUSTC) bar.rs
