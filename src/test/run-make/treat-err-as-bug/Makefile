-include ../tools.mk

all:
	$(RUSTC) err.rs -Z treat-err-as-bug 2>&1 \
	    | $(CGREP) "panicked at 'encountered error with \`-Z treat_err_as_bug'"
