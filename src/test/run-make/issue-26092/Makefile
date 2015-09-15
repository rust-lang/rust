-include ../tools.mk

all:
		$(BARE_RUSTC) -o "" blank.rs 2>&1 | \
			grep -i 'No such file or directory'
