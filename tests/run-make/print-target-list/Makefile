include ../tools.mk

# Checks that all the targets returned by `rustc --print target-list` are valid
# target specifications
all:
	for target in $(shell $(BARE_RUSTC) --print target-list); do \
		$(BARE_RUSTC) --target $$target --print sysroot; \
	done
