-include ../tools.mk

all:
	$(RUSTC) --crate-type=rlib crateA.rs
	$(RUSTC) --crate-type=rlib crateB.rs
	$(call REMOVE_RLIBS,crateA)
	# Ensure crateC fails to compile since dependency crateA is missing
	$(RUSTC) crateC.rs 2>&1 | \
		$(CGREP) 'can'"'"'t find crate for `crateA` which `crateB` depends on'
