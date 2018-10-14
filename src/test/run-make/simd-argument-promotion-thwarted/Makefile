-include ../../run-make-fulldeps/tools.mk

ifeq ($(TARGET),x86_64-unknown-linux-gnu)
all:
	$(RUSTC) t1.rs -C opt-level=3
	$(TMPDIR)/t1
	$(RUSTC) t2.rs -C opt-level=3
	$(TMPDIR)/t2
	$(RUSTC) t3.rs -C opt-level=3
	$(TMPDIR)/t3
else
all:
endif
