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

	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=diagnostics $(DEBUGINFOOPTS) --crate-type=lib --emit=metadata auxiliary/lib.rs
	! grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=diagnostics,object $(DEBUGINFOOPTS) --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=split-debuginfo $(DEBUGINFOOPTS) --crate-type=lib --emit=metadata auxiliary/lib.rs
	! grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1

    # FIXME: We should test the split debuginfo files, but we don't currently a good infra for that
	$(RUSTC) --remap-path-prefix $$PWD/auxiliary=/the/aux -Zremap-path-scope=split-debuginfo -Zunstable-options -Csplit-debuginfo=packed --crate-type=lib --emit=metadata auxiliary/lib.rs
	grep "/the/aux/lib.rs" $(TMPDIR)/liblib.rmeta || exit 1
	! grep "$$PWD/auxiliary" $(TMPDIR)/liblib.rmeta || exit 1
