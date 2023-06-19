# ignore-cross-compile
include ../tools.mk

all:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) my_lib.rs
	$(RUSTC) main.rs --test --extern my_lib=$(TMPDIR)/libmy_lib.rlib
