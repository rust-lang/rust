-include ../tools.mk

all:
	echo -e '\xD2' | $(RUSTC) - 2>&1 \
		| $(CGREP) "error: couldn't read from stdin, as it did not contain valid UTF-8"
