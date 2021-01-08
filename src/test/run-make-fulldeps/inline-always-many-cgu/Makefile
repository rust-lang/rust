-include ../tools.mk

all:
	$(RUSTC) foo.rs --emit llvm-ir -C codegen-units=2 -C opt-level=0
	if ![cat $(TMPDIR)/*.ll | $(CGREP) -e '\bcall\b']; then \
		echo "not found call instruction when one was expected"; \
		exit 1; \
	fi
	$(RUSTC) foo.rs --emit llvm-ir -C codegen-units=2 -C opt-level=1
	if cat $(TMPDIR)/*.ll | $(CGREP) -e '\bcall\b'; then \
		echo "found call instruction when one wasn't expected"; \
		exit 1; \
	fi
