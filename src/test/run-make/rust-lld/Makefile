# only-linux
# only-x86_64
# needs-rust-lld

# Ensure that the stable version of `-Zgcc-ld=lld` works:
# - `-C link-self-contained=linker`
# - `-C linker-flavor=gcc:lld`

-include ../../run-make-fulldeps/tools.mk
RUSTC_FLAGS = -C linker-flavor=gcc:lld -C link-self-contained=linker -Z unstable-options -C link-args=-Wl,-v

all:
	RUSTC_LOG=rustc_codegen_ssa::back::link=info $(RUSTC) $(RUSTC_FLAGS) empty.rs 2> $(TMPDIR)/output.txt
	$(CGREP) "lib/rustlib/x86_64-unknown-linux-gnu/bin/gcc-ld/ld" < $(TMPDIR)/output.txt
	$(CGREP) -e "^LLD [0-9]+\.[0-9]+\.[0-9]+" < $(TMPDIR)/output.txt
