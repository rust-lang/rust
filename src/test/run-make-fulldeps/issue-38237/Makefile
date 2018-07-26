-include ../tools.mk

ifeq ($(findstring stage1,$(RUST_BUILD_STAGE)),stage1)
# ignore stage1
all:

else
all:
	$(RUSTC) foo.rs; $(RUSTC) bar.rs
	$(RUSTDOC) baz.rs -L $(TMPDIR) -o $(TMPDIR)
endif
