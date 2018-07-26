-include ../tools.mk

all:
	$(RUSTC) --target x86_64_unknown-linux-musl main.rs 2>&1 | $(CGREP) \
		"error: Error loading target specification: Could not find specification for target"
