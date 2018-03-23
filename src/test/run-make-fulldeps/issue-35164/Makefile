-include ../tools.mk

all:
	$(RUSTC) main.rs --error-format json 2>&1 | $(CGREP) -e '"byte_start":490\b' '"byte_end":496\b'
