include ../../run-make-fulldeps/tools.mk

# ignore-msvc
# needs-rust-lld
all:
	RUSTC_LOG=rustc_codegen_ssa::back::link=info $(RUSTC) -Z gcc-ld=lld -C link-args=-Wl,-v main.rs 2> $(TMPDIR)/output.txt
	$(CGREP) -e "^LLD [0-9]+\.[0-9]+\.[0-9]+" < $(TMPDIR)/output.txt
