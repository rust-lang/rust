# ignore-windows

include ../tools.mk

all:
	echo 'mod unknown;' | $(RUSTC) --crate-type rlib - >$(TMPDIR)/unknown-mod.stdout 2>$(TMPDIR)/unknown-mod.stderr || echo "failed successfully"

# Bless like this: RUSTC_BLESS_TEST=1 ./x.py test tests/run-make/unknown-mod-stdin
ifdef RUSTC_BLESS_TEST
	cp "$(TMPDIR)"/unknown-mod.stdout unknown-mod.stdout
	cp "$(TMPDIR)"/unknown-mod.stderr unknown-mod.stderr
else
	$(DIFF) unknown-mod.stdout "$(TMPDIR)"/unknown-mod.stdout
	$(DIFF) unknown-mod.stderr "$(TMPDIR)"/unknown-mod.stderr
endif
