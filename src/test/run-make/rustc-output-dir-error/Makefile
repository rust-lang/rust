-include ../tools.mk

all:
	$(RUSTCNOARGS) foo.rs --out-dir ./baz/bar 2>&1 | grep "error: output directory baz/bar does not exist"
	$(RUSTCNOARGS) foo.rs --out-dir ./foo.rs 2>&1 | grep "error: specified output directory foo.rs is not a directory"
