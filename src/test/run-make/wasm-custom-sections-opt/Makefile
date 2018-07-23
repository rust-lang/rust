-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),wasm32-unknown-unknown)
all:
	$(RUSTC) foo.rs -O --target wasm32-unknown-unknown
	$(NODE) foo.js $(TMPDIR)/foo.wasm
else
all:
endif
