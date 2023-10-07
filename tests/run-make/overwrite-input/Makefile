include ../tools.mk

all:
	$(RUSTC) main.rs -o main.rs 2> $(TMPDIR)/file.stderr || echo "failed successfully"
	$(RUSTC) main.rs -o . 2> $(TMPDIR)/folder.stderr || echo "failed successfully"
	$(RUSTC_TEST_OP) "$(TMPDIR)"/file.stderr file.stderr
	$(RUSTC_TEST_OP) "$(TMPDIR)"/folder.stderr folder.stderr
