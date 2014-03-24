-include ../tools.mk

all:
	$(RUSTCNOARGS) foo.rs -o ./baz/bar.out 2>&1 | grep "error: specified output location's directory baz/bar.out does not exist"
	$(RUSTCNOARGS) foo.rs -o . 2>&1| grep "error: specified output location . is a directory"
