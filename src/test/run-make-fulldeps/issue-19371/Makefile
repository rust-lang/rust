include ../tools.mk

# This test ensures that rustc compile_input can be called twice in one task
# without causing a panic.
# The program needs the path to rustc to get sysroot.

all:
	$(RUSTC) foo.rs
	$(call RUN,foo $(TMPDIR) $(RUSTC))
