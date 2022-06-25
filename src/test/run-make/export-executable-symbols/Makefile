-include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) --crate-type=cdylib foo.rs --target $(TARGET)
	$(RUSTC) -Zexport-executable-symbols -lfoo -L "$(TMPDIR)" main.rs --target $(TARGET)
	$(call $(TMPDIR)/main)

