-include ../tools.mk

# The target used below doesn't support atomic operations. Verify that's the case
all:
	$(RUSTC) --print cfg --target thumbv6m-none-eabi | grep -qv target_has_atomic
