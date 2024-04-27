include ../tools.mk

all:
	$(RUSTC) --crate-type lib foo.rs -Z dump-mono-stats=$(TMPDIR) -Zdump-mono-stats-format=json
	cat $(TMPDIR)/foo.mono_items.json | $(CGREP) '"name":"bar"'
