-include ../tools.mk

all:
	$(RUSTC) main.rs --error-format json 2>&1 | grep -q '"byte_start":490.*"byte_end":496'
