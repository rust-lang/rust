# only-wasm32-bare

-include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) foo.rs -C lto -O --target wasm32-unknown-unknown
	wc -c < $(TMPDIR)/foo.wasm
	[ "`wc -c < $(TMPDIR)/foo.wasm`" -lt "25000" ]
