include ../tools.mk

all:
	cp non-utf8 $(TMPDIR)/non-utf.rs
	cat $(TMPDIR)/non-utf.rs | $(RUSTC) - 2>&1 \
		| $(CGREP) "error: couldn't read from stdin, as it did not contain valid UTF-8"
