-include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) --crate-type=cdylib foo.rs
	$(RUSTC) -Zexport-executable-symbols -lfoo -L "$(TMPDIR)" main.rs
	$(call $(TMPDIR)/main)

