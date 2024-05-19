include ../tools.mk

# This test ensures that rustc does not panic with `-o ""` option.

all:
	$(RUSTC) -o "" blank.rs 2>&1 | $(CGREP) -i 'panic' && exit 1 || exit 0
