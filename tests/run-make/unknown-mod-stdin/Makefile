# ignore-windows

include ../tools.mk

all:
	echo 'mod unknown;' | $(RUSTC) --crate-type rlib - >$(TMPDIR)/unknown-mod.stdout 2>$(TMPDIR)/unknown-mod.stderr || echo "failed successfully"
	$(RUSTC_TEST_OP) "$(TMPDIR)"/unknown-mod.stdout unknown-mod.stdout
	$(RUSTC_TEST_OP) "$(TMPDIR)"/unknown-mod.stderr unknown-mod.stderr
