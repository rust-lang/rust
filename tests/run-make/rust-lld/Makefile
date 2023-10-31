include ../tools.mk

# ignore-msvc
# needs-rust-lld
# ignore-s390x lld does not yet support s390x as target
all:
	RUSTC_LOG=rustc_codegen_ssa::back::link=info $(RUSTC) -Clink-self-contained=+linker -Clinker-flavor=gnu-lld-cc -Zunstable-options -Clink-args=-Wl,-v main.rs 2> $(TMPDIR)/output.txt
	$(CGREP) -e "^LLD [0-9]+\.[0-9]+\.[0-9]+" < $(TMPDIR)/output.txt
