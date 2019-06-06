-include ../tools.mk

all:
	$(RUSTC) foo.rs --emit llvm-ir -C codegen-units=2
	if cat $(TMPDIR)/*.ll | grep -v 'call void @llvm.sideeffect()' | $(CGREP) -e '\bcall\b'; then \
		echo "found call instruction when one wasn't expected"; \
		exit 1; \
	fi
