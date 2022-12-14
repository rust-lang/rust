include ../tools.mk

all:
	$(RUSTC) both.rs -C prefer-dynamic
	$(RUSTC) dylib.rs -C prefer-dynamic
	$(RUSTC) prog.rs
	$(call RUN,prog)
