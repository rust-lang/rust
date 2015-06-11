-include ../tools.mk

all:
		$(RUSTC) -o "" blank.rs 2>&1 | \
			grep 'No such file or directory'
