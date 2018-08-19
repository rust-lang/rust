-include ../tools.mk

ifeq ($(findstring stage1,$(RUST_BUILD_STAGE)),stage1)
# ignore stage1
all:

else
all:
	$(RUSTC) a.rs && $(RUSTC) b.rs
	$(BARE_RUSTC) c.rs -L dependency=$(TMPDIR) --extern b=$(TMPDIR)/libb.rlib \
		--out-dir=$(TMPDIR)
endif
