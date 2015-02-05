-include ../tools.mk

all:
	$(RUSTC) fakealloc.rs
	$(RUSTC) ../../../liballoc/lib.rs --cfg feature=\"external_crate\" --extern external=$(TMPDIR)/$(shell $(RUSTC) --print file-names fakealloc.rs)
