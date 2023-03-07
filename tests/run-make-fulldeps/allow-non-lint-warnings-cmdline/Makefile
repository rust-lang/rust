include ../tools.mk

# Test that -A warnings makes the 'empty trait list for derive' warning go away
OUT=$(shell $(RUSTC) foo.rs -A warnings 2>&1 | grep "warning" )

all: foo
	test -z '$(OUT)'

# This is just to make sure the above command actually succeeds
foo:
	$(RUSTC) foo.rs -A warnings
