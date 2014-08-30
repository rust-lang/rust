-include ../tools.mk

all:
	$(RUSTC) -o $(TMPDIR)/input.out --pretty expanded,hygiene input.rs
	cp input.pp.rs $(TMPDIR)/input.pp.rs

	# the name/ctxt numbers are very internals-dependent and thus
	# change relatively frequently, and testing for their exact values
	# them will fail annoyingly, so we just check their positions
	# (using a non-constant replacement like this will make it less
	# likely the compiler matches whatever other dummy value we
	# choose).
	sed -i "s/[0-9]*#[0-9]*/$(shell date)/g" $(TMPDIR)/input.out $(TMPDIR)/input.pp.rs

	diff -u $(TMPDIR)/input.out $(TMPDIR)/input.pp.rs
