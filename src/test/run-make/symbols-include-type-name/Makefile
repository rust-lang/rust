-include ../tools.mk

# Check that symbol names for methods include type names, instead of <impl>.

OUT=$(TMPDIR)/lib.s

all:
	$(RUSTC) --crate-type staticlib --emit asm lib.rs
	grep Def $(OUT)
