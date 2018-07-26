-include ../tools.mk

# Issue #10814
#
# these are no_std to avoid having to have the standard library or any
# linkers/assemblers for the relevant platform

all:
	$(RUSTC) foo.rs --target=i686-unknown-linux-gnu
	$(RUSTC) bar.rs --target=x86_64-unknown-linux-gnu 2>&1 \
		| $(CGREP) 'couldn'"'"'t find crate `foo` with expected target triple x86_64-unknown-linux-gnu'
