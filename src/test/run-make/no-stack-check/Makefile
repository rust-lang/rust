-include ../tools.mk

all:
	$(RUSTC) -O --emit asm attr.rs
	! grep -q morestack $(TMPDIR)/attr.s
	$(RUSTC) -O --emit asm flag.rs
	grep -q morestack $(TMPDIR)/flag.s
	$(RUSTC) -O --emit asm -C no-stack-check flag.rs
	! grep -q morestack $(TMPDIR)/flag.s
