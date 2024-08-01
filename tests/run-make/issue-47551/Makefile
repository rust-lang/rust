# only-linux
# ignore-32bit

include ../tools.mk

all:
	# --target $(TARGET) ensures the right gcc flags are used for cross compilation
	$(RUSTC) --target $(TARGET) eh_frame-terminator.rs
	$(call RUN,eh_frame-terminator) | $(CGREP) '1122334455667788'
	objdump --dwarf=frames $(TMPDIR)/eh_frame-terminator | $(CGREP) 'ZERO terminator'
