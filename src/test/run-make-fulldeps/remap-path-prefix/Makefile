-include ../tools.mk

# ignore-windows

# Checks if remapping works if the remap-from string contains path to the working directory plus more
all:
	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1
	$(RUSTDOC) --test --no-run -C debuginfo=1 -C split-debuginfo=off -Zunstable-options --remap-path-prefix $$PWD/auxiliary=/the/aux --remap-path-prefix $$PWD=/the/pwd --persist-doctests=$(TMPDIR)/rustdoc-remap auxiliary/lib.rs
	grep "/the/pwd" $(TMPDIR)/rustdoc-remap/auxiliary_lib_rs_1_0/rust_out || exit 1
	! grep "$$PWD" $(TMPDIR)/rustdoc-remap/auxiliary_lib_rs_1_0/rust_out || exit 1
