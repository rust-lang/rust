-include ../tools.mk

all:
	# Let's get a nice error message
	$(RUSTC) foo.rs --dep-info foo/bar/baz 2>&1 | \
		grep "error writing dependencies"
	# Make sure the filename shows up
	$(RUSTC) foo.rs --dep-info foo/bar/baz 2>&1 | grep "baz"
