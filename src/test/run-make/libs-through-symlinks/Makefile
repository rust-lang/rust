-include ../tools.mk

ifdef IS_WINDOWS
all:
else

NAME := $(shell $(RUSTC) --crate-file-name foo.rs)

all:
	mkdir -p $(TMPDIR)/outdir
	$(RUSTC) foo.rs -o $(TMPDIR)/outdir/$(NAME)
	ln -nsf outdir/$(NAME) $(TMPDIR)
	RUST_LOG=rustc::metadata::loader $(RUSTC) bar.rs
endif
