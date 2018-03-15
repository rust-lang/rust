-include ../tools.mk

ifeq ($(findstring stage1,$(RUST_BUILD_STAGE)),stage1)
# ignore stage1
all:

else
all:
	$(RUSTC) foo.rs
	$(RUSTC) bar.rs --emit dep-info
	grep "proc-macro source" $(TMPDIR)/bar.d && exit 1 || exit 0
endif
