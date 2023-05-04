# needs-unwind -Cpanic=abort gives different MIR output
include ../tools.mk

all:
	$(RUSTC) main.rs --emit=mir -o "$(TMPDIR)"/dump.mir

ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/dump.mir dump.mir
else
	$(DIFF) dump.mir "$(TMPDIR)"/dump.mir
endif
