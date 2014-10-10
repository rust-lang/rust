-include ../tools.mk

ifndef IS_WINDOWS
all:
	$(RUSTC) -O --emit asm attr.rs
	! grep -q morestack $(TMPDIR)/attr.s
	$(RUSTC) -O --emit asm flag.rs
	grep -q morestack $(TMPDIR)/flag.s
	$(RUSTC) -O --emit asm -C no-stack-check flag.rs
	! grep -q morestack $(TMPDIR)/flag.s
else
# On Windows we use __chkstk and it only appears in functions with large allocations,
# so this test wouldn't be reliable.
all:
endif
