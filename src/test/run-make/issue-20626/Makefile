-include ../tools.mk

# Test output to be four
# The original error only occurred when printing, not when comparing using assert!

all:
	$(RUSTC) foo.rs -O
	[ `$(call RUN,foo)` = "4" ]
