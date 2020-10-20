-include ../../run-make-fulldeps/tools.mk

NM=nm

all: main.rs
	$(RUSTC) $< -O
	$(NM) $(call RUN_BINFILE,main) | $(CGREP) -v panicking panic_fmt panic_bounds_check pad_integral Display Debug
