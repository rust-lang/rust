-include ../tools.mk

all:
	$(RUSTC) ep-lib.rs
	$(RUSTC) ep-vec.rs

	$(RUSTC) basic.rs --extern ep_lib=$(TMPDIR)/libep_lib.rlib
	$(RUSTC) shadow-mod.rs --extern ep_lib=$(TMPDIR)/libep_lib.rlib
	$(RUSTC) shadow-prelude.rs --extern Vec=$(TMPDIR)/libep_vec.rlib
	$(RUSTC) feature-gate.rs --extern ep_lib=$(TMPDIR)/libep_lib.rlib 2>&1 | $(CGREP) "access to extern crates through prelude is experimental"
	$(RUSTC) relative-only.rs --extern ep_lib=$(TMPDIR)/libep_lib.rlib 2>&1 | $(CGREP) "unresolved import"
	$(RUSTC) relative-only.rs --extern ep_lib=$(TMPDIR)/libep_lib.rlib 2>&1 | $(CGREP) "failed to resolve"
