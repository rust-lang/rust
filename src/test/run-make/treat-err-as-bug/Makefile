-include ../tools.mk

all:
	$(RUSTC) err.rs -Z treat-err-as-bug 2>&1 \
	    | grep -q "panicked at 'encountered error with .-Z treat_err_as_bug'"
