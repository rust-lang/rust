-include ../tools.mk

all:
	echo 'fn main(){}' | $(RUSTC) -Z no-link -
	$(RUSTC) -Z link-only $(TMPDIR)/rust_out.rlink
	$(call RUN,rust_out)
