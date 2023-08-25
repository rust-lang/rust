include ../tools.mk

# ignore-windows

# Checks if remapping works if the remap-from string contains path to the working directory plus more
all:
	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=object --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=diagnostics --crate-type=lib --emit=metadata auxiliary/lib.rs
	! grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=diagnostics,object --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=split-debuginfo --crate-type=lib --emit=metadata auxiliary/lib.rs
	! grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1
