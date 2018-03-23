-include ../tools.mk

REPLACEMENT := s/[0-9][0-9]*\#[0-9][0-9]*/$(shell date)/g

all:
	$(RUSTC) -o $(TMPDIR)/input.out -Z unstable-options \
		--pretty expanded,hygiene input.rs

	# the name/ctxt numbers are very internals-dependent and thus
	# change relatively frequently, and testing for their exact values
	# them will fail annoyingly, so we just check their positions
	# (using a non-constant replacement like this will make it less
	# likely the compiler matches whatever other dummy value we
	# choose).
	#
	# (These need to be out-of-place because OSX/BSD & GNU sed
	# differ.)
	sed "$(REPLACEMENT)" input.pp.rs > $(TMPDIR)/input.pp.rs
	sed "$(REPLACEMENT)" $(TMPDIR)/input.out > $(TMPDIR)/input.out.replaced

	diff -u $(TMPDIR)/input.out.replaced $(TMPDIR)/input.pp.rs
