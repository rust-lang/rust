-include ../tools.mk

# Checks if remapping works if the remap-from string contains path to the working directory plus more
all:
	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux --crate-type=lib --emit=metadata auxiliary/lib.rs
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1
