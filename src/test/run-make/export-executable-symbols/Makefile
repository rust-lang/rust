-include ../../run-make-fulldeps/tools.mk

all:
	$(RUSTC) --edition=2018 --crate-type=cdylib foo.rs -o $(TMPDIR)/libfoo.so
	$(RUSTC) --edition=2018 -Zexport-executable-symbols -lfoo -L $(TMPDIR) main.rs -o $(TMPDIR)/main
	$(call $(TMPDIR)/main)

