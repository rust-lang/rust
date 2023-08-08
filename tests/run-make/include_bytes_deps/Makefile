include ../tools.mk

# ignore-freebsd

all:
	$(RUSTC) --emit dep-info main.rs
	$(CGREP) "input.txt" "input.bin" "input.md" < $(TMPDIR)/main.d
