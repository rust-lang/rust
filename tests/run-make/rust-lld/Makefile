include ../tools.mk

# ignore-msvc
# needs-rust-lld
# ignore-s390x lld does not yet support s390x as target
all:
	RUSTC_LOG=rustc_codegen_ssa::back::link=info $(RUSTC) -Clink-self-contained=+linker -Zlinker-features=+lld -Zunstable-options -Clink-args=-Wl,-v main.rs 2> $(TMPDIR)/output.txt
	$(CGREP) -e "^LLD [0-9]+\.[0-9]+\.[0-9]+" < $(TMPDIR)/output.txt

	# while we're here, also check that the last linker feature flag "wins"
	RUSTC_LOG=rustc_codegen_ssa::back::link=info $(RUSTC) -Clink-self-contained=+linker -Zlinker-features=-lld -Zlinker-features=+lld -Zunstable-options -Clink-args=-Wl,-v main.rs 2> $(TMPDIR)/output.txt
	$(CGREP) -e "^LLD [0-9]+\.[0-9]+\.[0-9]+" < $(TMPDIR)/output.txt
