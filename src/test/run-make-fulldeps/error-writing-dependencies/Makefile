-include ../tools.mk

all:
	# Let's get a nice error message
	$(BARE_RUSTC) foo.rs --emit dep-info --out-dir foo/bar/baz 2>&1 | \
		$(CGREP) "error writing dependencies"
	# Make sure the filename shows up
	$(BARE_RUSTC) foo.rs --emit dep-info --out-dir foo/bar/baz 2>&1 | $(CGREP) "baz"
