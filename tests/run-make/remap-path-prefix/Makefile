include ../tools.mk

# ignore-windows

ifeq ($(UNAME),Darwin)
  DEBUGINFOOPTS := -Csplit-debuginfo=off
else
  DEBUGINFOOPTS :=
endif

all: remap remap-with-scope

# Checks if remapping works if the remap-from string contains path to the working directory plus more
remap:
	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

remap-with-scope:
	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=object $(DEBUGINFOOPTS) --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=macro $(DEBUGINFOOPTS) --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=diagnostics,object $(DEBUGINFOOPTS) --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1
