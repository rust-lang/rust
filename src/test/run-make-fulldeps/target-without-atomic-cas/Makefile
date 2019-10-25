-include ../tools.mk

# The target used below doesn't support atomic CAS operations. Verify that's the case
all:
	$(RUSTC) --print cfg --target thumbv6m-none-eabi | $(CGREP) -v 'target_has_atomic="cas"'
