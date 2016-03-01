-include ../tools.mk

all:
	-mkdir -p $(TMPDIR)/good
	-mkdir -p $(TMPDIR)/bad
	$(BARE_RUSTC) ./good/lib.rs -C prefer-dynamic --out-dir="$(TMPDIR)/good"
	$(BARE_RUSTC) ./bad/lib.rs -C prefer-dynamic --out-dir="$(TMPDIR)/bad"
	$(BARE_RUSTC) -L "$(TMPDIR)/good" -C prefer-dynamic -Crpath ./main.rs --out-dir="$(TMPDIR)"
	# This should succeed because the correct library is in LD_LIBRARY_PATH
	$(LD_LIB_PATH_ENVVAR)="$(TMPDIR)/good:$($(LD_LIB_PATH_ENVVAR))" $(TMPDIR)/main
	# This should fail because the wrong library is in LD_LIBRARY_PATH
	OUTPUT=`$(LD_LIB_PATH_ENVVAR)="$(TMPDIR)/bad:$($(LD_LIB_PATH_ENVVAR))" $(TMPDIR)/main || exit 0`
	if ["$(OUTPUT)" == "bad"]; then exit 1; fi
