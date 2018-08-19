-include ../tools.mk

# Test that -A warnings makes the 'empty trait list for derive' warning go away
DEP=$(shell $(RUSTC) bar.rs)
OUT=$(shell $(RUSTC) foo.rs -A warnings 2>&1 | grep "warning" )

all: foo bar
	test -z '$(OUT)'

# These are just to ensure that the above commands actually work
bar:
	$(RUSTC) bar.rs

foo: bar
	$(RUSTC) foo.rs -A warnings
