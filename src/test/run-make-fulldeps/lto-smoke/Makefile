-include ../tools.mk

all:
	$(RUSTC) lib.rs
	$(RUSTC) main.rs -C lto
	$(call RUN,main)
