# needs-unwind -Cpanic=abort gives different MIR output
include ../tools.mk

all:
	$(RUSTC) main.rs --emit=mir -o "$(TMPDIR)"/dump.mir
	$(RUSTC_TEST_OP) "$(TMPDIR)"/dump.mir dump.mir
