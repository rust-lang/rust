# ignore-cross-compile
# needs-unwind
include ../tools.mk

all:
	$(RUSTC) lib.rs
	$(RUSTC) main.rs
	$(call RUN,main)
