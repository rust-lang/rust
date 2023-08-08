include ../tools.mk

all:
	$(RUSTC) --emit=metadata --crate-type lib stable.rs
	$(RUSTC) --emit=metadata --extern stable=$(TMPDIR)/libstable.rmeta main.rs 2>&1 >/dev/null \
		| $(CGREP) -e "stable since $$(cat $(S)/src/version)(-[a-zA-Z]+)?"
