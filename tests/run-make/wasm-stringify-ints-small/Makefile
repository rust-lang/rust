include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),wasm32-unknown-unknown)
all:
	$(RUSTC) foo.rs -C lto -O --target wasm32-unknown-unknown
	wc -c < $(TMPDIR)/foo.wasm
	[ "`wc -c < $(TMPDIR)/foo.wasm`" -lt "25000" ]
else
all:
endif
