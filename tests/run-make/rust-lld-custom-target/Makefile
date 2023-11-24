include ../tools.mk

# needs-rust-lld
# only-x86_64-unknown-linux-gnu
all:
	RUSTC_LOG=rustc_codegen_ssa::back::link=info $(RUSTC) --crate-type cdylib --target custom-target.json -Clink-args=-Wl,-v lib.rs 2> $(TMPDIR)/output.txt
	$(CGREP) -e "^LLD [0-9]+\.[0-9]+\.[0-9]+" < $(TMPDIR)/output.txt
