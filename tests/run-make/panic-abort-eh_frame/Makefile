# only-linux
#
# This test ensures that `panic=abort` code (without `C-unwind`, that is) should not have any
# unwinding related `.eh_frame` sections emitted.

include ../tools.mk

all:
	$(RUSTC) foo.rs --crate-type=lib --emit=obj=$(TMPDIR)/foo.o -Cpanic=abort
	objdump --dwarf=frames $(TMPDIR)/foo.o | $(CGREP) -v 'DW_CFA'
