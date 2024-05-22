# This test checks that manually setting the output file as a bare file with no file extension still results in successful compilation.

# ignore-cross-compile
include ../tools.mk

all:
	cp foo.rs $(TMPDIR)
	cd $(TMPDIR) && $(RUSTC) -o foo foo.rs
	$(call RUN,foo)
