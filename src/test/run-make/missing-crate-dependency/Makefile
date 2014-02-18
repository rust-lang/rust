-include ../tools.mk

all: 
	$(RUSTC) --crate-type=rlib crateA.rs
	$(RUSTC) --crate-type=rlib crateB.rs
	rm $(TMPDIR)/$(call RLIB_GLOB,crateA)
	# Ensure crateC fails to compile since dependency crateA is missing
	$(RUSTC) crateC.rs 2>&1 | \
		grep "error: can't find crate for \`crateA\` which \`crateB\` depends on"
