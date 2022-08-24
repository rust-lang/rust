include ../../run-make-fulldeps/tools.mk

# only-wasm32-bare

all:
	$(RUSTC) foo.rs -C lto -O --target wasm32-unknown-unknown --cfg a
	wc -c < $(TMPDIR)/foo.wasm
	[ "`wc -c < $(TMPDIR)/foo.wasm`" -lt "1024" ]
	$(RUSTC) foo.rs -C lto -O --target wasm32-unknown-unknown --cfg b
	wc -c < $(TMPDIR)/foo.wasm
	[ "`wc -c < $(TMPDIR)/foo.wasm`" -lt "5120" ]
	$(RUSTC) foo.rs -C lto -O --target wasm32-unknown-unknown --cfg c
	wc -c < $(TMPDIR)/foo.wasm
	[ "`wc -c < $(TMPDIR)/foo.wasm`" -lt "5120" ]
	$(RUSTC) foo.rs -C lto -O --target wasm32-unknown-unknown --cfg d
	wc -c < $(TMPDIR)/foo.wasm
	[ "`wc -c < $(TMPDIR)/foo.wasm`" -lt "5120" ]
