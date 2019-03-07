-include ../tools.mk

all:
	$(RUSTC) err.rs -Z treat-err-as-bug 2>&1 \
	    | $(CGREP) "panicked at 'aborting due to \`-Z treat-err-as-bug=1\`'"
