-include ../tools.mk

# check that the compile generated symbols for strings, binaries,
# vtables, etc. have semisane names (e.g. `str.1234`); it's relatively
# easy to accidentally modify the compiler internals to make them
# become things like `str"str"(1234)`.

OUT=$(TMPDIR)/lib.s

all:
	$(RUSTC) lib.rs --emit=asm --crate-type=staticlib
	# just check for symbol declarations with the names we're expecting.
	grep 'str.[0-9][0-9]*:' $(OUT)
	grep 'byte_str.[0-9][0-9]*:' $(OUT)
	grep 'vtable.[0-9][0-9]*' $(OUT)
