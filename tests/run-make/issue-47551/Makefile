# only-linux
# ignore-32bit

include ../tools.mk

all:
	$(RUSTC) eh_frame-terminator.rs
	$(call RUN,eh_frame-terminator) | $(CGREP) '1122334455667788'
	objdump --dwarf=frames $(TMPDIR)/eh_frame-terminator | $(CGREP) 'ZERO terminator'
