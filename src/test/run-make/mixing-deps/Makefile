-include ../tools.mk

all:
	$(RUSTC) both.rs
	$(RUSTC) dylib.rs -Z prefer-dynamic
	$(RUSTC) prog.rs
	$(call RUN,prog)
