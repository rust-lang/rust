# This test is designed to intentionally introduce a circular dependency scenario to check that a specific compiler bug doesn't make a resurgence.
# The bug in question arose when at least one crate required a global allocator, and that crate was placed after the one defining it in the linker order. 
# The generated symbols.o should not result in any linker errors.
# See https://github.com/rust-lang/rust/issues/112715

# ignore-cross-compile
include ../tools.mk

all:
	rm -rf $(TMPDIR) && mkdir $(TMPDIR)
	$(RUSTC) my_lib.rs
	$(RUSTC) main.rs --test --extern my_lib=$(TMPDIR)/libmy_lib.rlib
