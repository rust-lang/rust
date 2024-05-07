# This test intentionally creates a circular dependency, and checks if this causes the resurgence of the compiler bug linked below.
# See https://github.com/rust-lang/rust/issues/112715

# ignore-cross-compile
include ../tools.mk

all:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) my_lib.rs
	$(RUSTC) main.rs --test --extern my_lib=$(TMPDIR)/libmy_lib.rlib
