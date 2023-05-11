# ignore-cross-compile
# ignore-i686-pc-windows-gnu
# needs-unwind

# This test doesn't work on 32-bit MinGW as cdylib has its own copy of unwinder
# so cross-DLL unwinding does not work.

include ../tools.mk

all:
	$(RUSTC) bar.rs --crate-type=cdylib
	$(RUSTC) foo.rs
	$(call RUN,foo) 2>&1 | $(CGREP) "Rust cannot catch foreign exceptions"
