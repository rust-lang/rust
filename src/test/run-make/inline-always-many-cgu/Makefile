-include ../tools.mk

all:
	$(RUSTC) foo.rs --emit llvm-ir -C codegen-units=2
	if grep -w call $(TMPDIR)/*.ll; then \
		echo "found call instruction when one wasn't expected"; \
		exit 1; \
	fi
