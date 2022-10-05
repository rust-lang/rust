include ../tools.mk

all:
	$(RUSTC) bar.rs --crate-type=cdylib
	$(RUSTC) foo.rs
	$(call RUN,foo) 2>&1 | $(CGREP) "Rust cannot catch foreign exceptions"
