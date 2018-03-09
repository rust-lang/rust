-include ../tools.mk

all:
	$(RUSTC) -o "" blank.rs 2>&1 | $(CGREP) -i 'No such file or directory'
