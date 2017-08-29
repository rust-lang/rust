-include ../tools.mk

ifeq ($(findstring stage1,$(RUST_BUILD_STAGE)),stage1)
# ignore stage1
all:

else
all:
	$(RUSTC) a.rs && $(RUSTC) b.rs && $(RUSTC) c.rs
endif
