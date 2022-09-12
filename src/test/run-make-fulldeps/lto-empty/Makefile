include ../tools.mk

all: cdylib-fat cdylib-thin

cdylib-fat:
	$(RUSTC) lib.rs -C lto=fat -C opt-level=3 -C incremental=$(TMPDIR)/inc-fat
	$(RUSTC) lib.rs -C lto=fat -C opt-level=3 -C incremental=$(TMPDIR)/inc-fat

cdylib-thin:
	$(RUSTC) lib.rs -C lto=thin -C opt-level=3 -C incremental=$(TMPDIR)/inc-thin
	$(RUSTC) lib.rs -C lto=thin -C opt-level=3 -C incremental=$(TMPDIR)/inc-thin

