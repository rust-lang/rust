-include ../tools.mk

all:
	$(RUSTC) foo.rs --crate-type staticlib
	$(RUSTC) bar.rs 2>&1 | $(CGREP) "found staticlib"
