-include ../tools.mk

all:
	$(RUSTC) lib.rs
	$(RUSTC) main.rs -Z lto
	$(call RUN,main)
