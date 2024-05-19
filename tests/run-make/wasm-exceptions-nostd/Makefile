include ../tools.mk

# only-wasm32-bare

# Add a few command line args to make exceptions work
RUSTC := $(RUSTC) -C llvm-args=-wasm-enable-eh
RUSTC := $(RUSTC) -C target-feature=+exception-handling
RUSTC := $(RUSTC) -C panic=unwind

all:
	$(RUSTC) src/lib.rs --target wasm32-unknown-unknown
	$(NODE) verify.mjs $(TMPDIR)/lib.wasm
