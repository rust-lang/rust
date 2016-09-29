# Test that we can link, using the embedded lld, a "bare-metal" executable for a foreign
# architecture (ARM Cortex M, in this case) without relying on a external linker (like
# arm-none-eabi-gcc) or external C stuff (libraries or startup objects).

# TODO this test should be skipped when lld support has not been built into rustc

-include ../tools.mk

all:
	$(RUSTC) --target cortex-m3 -Z print-link-args -Z use-lld bare.rs
