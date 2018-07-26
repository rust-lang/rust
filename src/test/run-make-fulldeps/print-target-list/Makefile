-include ../tools.mk

# Checks that all the targets returned by `rustc --print target-list` are valid
# target specifications
# TODO remove the '*ios*' case when rust-lang/rust#29812 is fixed
all:
	for target in $(shell $(BARE_RUSTC) --print target-list); do \
		case $$target in \
			*ios*) \
				;; \
			*) \
				$(BARE_RUSTC) --target $$target --print sysroot \
				;; \
			esac \
	done
